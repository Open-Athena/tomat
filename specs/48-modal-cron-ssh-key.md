# Spec 48 — Modal `tomat-iris-sync-cron` SSH-key follow-up

## Symptom
Every Modal cron tick fails with:
```
Could not connect to controller: SSH key not found at
/root/.ssh/google_compute_engine. gcloud compute ssh requires this
key to connect to VMs.
```
Raised by `iris.cluster.providers.gcp.controller._check_gcloud_ssh_key`
(`.venv/.../iris/cluster/providers/gcp/controller.py:148`). The Modal
container's filesystem has neither `~/.ssh/google_compute_engine` nor
its `.pub`, so the controller-tunnel setup short-circuits before any
RPC is dispatched.

## Why it's needed at all
`iris job list --json` (used by `_sync()` for each of `/ryan/tomat`,
`/ryan/train`, `/ryan/eval`) calls the controller's gRPC `ListJobs`
endpoint. The controller VM is reachable only via an IAP `gcloud
compute ssh` tunnel (see `_build_tunnel_ssh_cmd`,
`.../gcp/controller.py:192-224`), which spawns:
```
gcloud compute ssh <controller-vm> --project=... --zone=...
    --ssh-key-file=~/.ssh/google_compute_engine -- -L 127.0.0.1:PORT:...
```
That ssh process needs both the private key (key authentication on the
underlying ssh client) AND the key's public half registered in either
GCE project metadata or OS Login. The GCE cron VM works because
`setup-iris-cron-vm.sh:68-72` `ssh-keygen`s the key on first setup, and
`gcloud`'s first call auto-registers the `.pub` against the cron VM's
project — but Marin's controller lives in `hai-gcp-models`, NOT
`oa-internal-450019`, so the auto-registration path may or may not
extend to it. In practice my laptop's key works against Marin's
controller because I'm an OS-Login-registered user on `hai-gcp-models`
via `iris-worker@hai-gcp-models.iam.gserviceaccount.com`.

## Options

### A. Ship the laptop SSH key as a Modal secret (recommended)
Mount the existing local key pair into the container before iris runs:
```bash
modal secret create marin-iris-ssh \
    GCE_SSH_PRIVATE="$(cat ~/.ssh/google_compute_engine)" \
    GCE_SSH_PUBLIC="$(cat ~/.ssh/google_compute_engine.pub)"
```
Then in `scripts/cron_iris_sync_modal.py` `_materialize_adc()` (rename
`_materialize_creds`) write both files with `0600`/`0644` perms before
the iris subprocess starts. Add the secret to the function decorators:
```python
ssh_secret = modal.Secret.from_name("marin-iris-ssh")
# ...
@app.function(... secrets=[adc_secret, r2_secret, ssh_secret])
```
Pros:
- Reuses the same identity that already works from the laptop, so no
  new OS Login / project-metadata registration needed.
- Smallest delta: one secret + a few lines of file IO.
- Mirrors the GCE VM's working setup (same key, just delivered through
  Modal secrets instead of `ssh-keygen` on the VM).

Cons:
- Ties Modal cron to ryan's personal identity. If the key rotates or
  ryan leaves OA, the cron breaks. (Long-term fix is a Marin-issued SA
  with read-only RPC access — see Option C.)
- Private key material in a Modal secret. Modal secrets are
  encrypted-at-rest with per-workspace KMS, scoped to this app's
  functions — comparable to the ADC token already stored in
  `marin-iris-adc`, so no new sensitivity tier.

### B. ssh-keygen inside the container
Add `apt_install("openssh-client")` + a startup-time `ssh-keygen -t rsa
-f ~/.ssh/google_compute_engine -N ''` step. Does NOT work: the
resulting public key is brand-new, hasn't been registered with OS
Login on `hai-gcp-models`, and gcloud's auto-registration path
requires interactive auth (or a project-owner SA, which we don't have).
The tunnel would fail at the ssh handshake instead of the
`_check_gcloud_ssh_key` precheck — same outcome, harder to diagnose.

Not recommended.

### C. Skip the iris RPC entirely
Two sub-options:
1. **Talk to the controller's gRPC endpoint directly.** Requires
   knowing the controller's IP + a non-tunneled port (currently
   tunnel-only) + a token. Big lift; effectively requires Marin to
   expose a public read-only endpoint.
2. **Pull job state from an alternate source.** E.g. parse the wandb
   runs Modal already syncs, or read GCS log artifacts. Coarser
   (would miss `task_state_counts`, exit codes), defeats the purpose
   of spec 45.

Both Cs are >1-day rewrites and would diverge from the canonical
`tomat iris sync` path. Not recommended without a Marin ops change.

## Recommendation
**Option A.** Lowest delta, mirrors the working GCE VM identity, keeps
the Modal cron schema-aligned with `tomat iris sync` and
`scripts/iris-sync.py`. Should take ~15 minutes of work + one redeploy
to verify.

Until Option A lands, the GCE cron at `tomat-iris-cron` (us-east1-d)
continues to drive `iris-state.json` every minute (with the fresh
`task_state_counts` field after the spec-48 sibling fix to
`scripts/iris-sync.py`). The Modal cron is dormant-but-broken; it
re-uploads nothing because `_sync()` short-circuits on
`payload["count"] == 0`. No production impact while it sits idle, but
the failure logs will keep accumulating in Modal until the secret is
added or the cron is paused.

## Out of scope
- Long-term: get a Marin-issued read-only SA with permission to call
  `JobService.ListJobs`, swap both ADC + SSH secrets for that one
  identity, decommission the GCE VM. Track separately — depends on a
  Marin ops ask.
- Single-source-of-truth between `scripts/iris-sync.py`,
  `scripts/cron_iris_sync_modal.py`, and `tomat iris sync` (currently
  three near-identical `_build_payload` functions). Factor when the
  schema changes a second time.
