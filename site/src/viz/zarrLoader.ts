// Lightweight zarr v3 level-0 reader.
//
// Tomat's voxel zarrs are written by `marin.voxel_zarr.write_voxel_zarr`:
// each `<root>.zarr/<level>/` is a single-chunk float16 + zstd array. We
// don't need zarrita here — the schema is fixed enough that fetching
// `<root>.zarr/0/zarr.json` + `<root>.zarr/0/c/0/0/0`, decompressing with
// fzstd, and decoding fp16 → fp32 covers every grid in the dataset.
//
// (zarrita 0.7.2 doesn't ship a zstd codec or a float16 cast value, so
// going DIY here actually saves bytes + bundle complexity.)

import { decompress as zstdDecompress } from 'fzstd'

export interface ZarrLevelMeta {
  shape: [number, number, number]
  dtype: string
  chunkShape: [number, number, number]
  codecs: Array<{ name: string; configuration?: Record<string, unknown> }>
  /** `little` / `big` — endianness declared by the `bytes` codec. */
  endian: 'little' | 'big'
}

export interface LoadedLevel {
  meta: ZarrLevelMeta
  /** Decoded float32 values, C-order (z, y, x) — same layout the writer
   *  used; downstream is just a 1-D scan, so layout barely matters. */
  data: Float32Array
}

/** Fetch + decode level 0 of a zarr v3 group. The url is the .zarr root
 *  (no trailing slash). Throws on any unexpected codec / dtype to surface
 *  schema drift early rather than silently returning bad data. */
export async function loadZarrLevel0(zarrUrl: string): Promise<LoadedLevel> {
  const metaRes = await fetch(`${zarrUrl}/0/zarr.json`)
  if (!metaRes.ok) {
    throw new Error(`fetch ${zarrUrl}/0/zarr.json: ${metaRes.status}`)
  }
  const raw = await metaRes.json() as {
    shape: number[]
    data_type: string
    chunk_grid: { configuration: { chunk_shape: number[] } }
    codecs: Array<{ name: string; configuration?: Record<string, unknown> }>
  }
  if (raw.shape.length !== 3) {
    throw new Error(`expected 3-D shape, got ${JSON.stringify(raw.shape)}`)
  }
  if (raw.data_type !== 'float16') {
    throw new Error(`expected data_type=float16, got ${raw.data_type}`)
  }
  const codecs = raw.codecs ?? []
  const bytesCodec = codecs.find(c => c.name === 'bytes')
  const endian = (bytesCodec?.configuration?.endian as string | undefined) ?? 'little'
  if (endian !== 'little' && endian !== 'big') {
    throw new Error(`unexpected endian: ${endian}`)
  }
  const meta: ZarrLevelMeta = {
    shape: raw.shape as [number, number, number],
    dtype: raw.data_type,
    chunkShape: raw.chunk_grid.configuration.chunk_shape as [number, number, number],
    codecs,
    endian,
  }
  // Confirm the array is a single chunk (the writer's invariant for
  // grids ≤200³). If we ever start chunking, the reader needs to fan out
  // over c/<i>/<j>/<k> — caller will see the explicit error below.
  if (
    meta.chunkShape[0] !== meta.shape[0] ||
    meta.chunkShape[1] !== meta.shape[1] ||
    meta.chunkShape[2] !== meta.shape[2]
  ) {
    throw new Error(
      `multi-chunk arrays not supported (shape=${meta.shape}, chunk=${meta.chunkShape})`,
    )
  }
  const chunkRes = await fetch(`${zarrUrl}/0/c/0/0/0`)
  if (!chunkRes.ok) {
    throw new Error(`fetch ${zarrUrl}/0/c/0/0/0: ${chunkRes.status}`)
  }
  // Buffer type annotation: TS narrows `new Uint8Array(ArrayBuffer)` to
  // `Uint8Array<ArrayBuffer>`, but fzstd's decompress returns
  // `Uint8Array<ArrayBufferLike>` (it can target a SharedArrayBuffer in
  // worker contexts). The wider type round-trips fine through our DataView.
  let buf: Uint8Array<ArrayBufferLike> = new Uint8Array(await chunkRes.arrayBuffer())
  // Apply codecs in reverse order: encode pipeline is [bytes, zstd], so
  // decode is [zstd, bytes]. The reverse-iterate is robust to additional
  // optional codecs (e.g. crc32c) if they're ever added.
  for (let i = codecs.length - 1; i >= 0; i--) {
    const c = codecs[i]
    if (c.name === 'zstd') {
      buf = zstdDecompress(buf)
    } else if (c.name === 'bytes') {
      // bytes codec is a no-op for our purposes (raw byte stream → typed array);
      // endian conversion handled below.
    } else {
      throw new Error(`unsupported codec: ${c.name}`)
    }
  }
  const n = meta.shape[0] * meta.shape[1] * meta.shape[2]
  if (buf.byteLength !== n * 2) {
    throw new Error(
      `chunk size mismatch: got ${buf.byteLength} bytes, expected ${n * 2} (n=${n} × fp16)`,
    )
  }
  const data = decodeFloat16(buf, n, endian === 'little')
  return { meta, data }
}

/** Decode an IEEE-754 fp16 byte stream into a Float32Array. Implements the
 *  fp16→fp32 cast inline so we don't pull in an extra dependency just for
 *  one ~10-line function. Handles subnormals + NaN + ±Inf. */
function decodeFloat16(bytes: Uint8Array, n: number, littleEndian: boolean): Float32Array {
  const out = new Float32Array(n)
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  for (let i = 0; i < n; i++) {
    const u16 = dv.getUint16(i * 2, littleEndian)
    out[i] = fp16ToFp32(u16)
  }
  return out
}

/** IEEE-754 binary16 → binary32 conversion. Branchy but only runs ~400k
 *  times per load (single-host), so it's fine. */
function fp16ToFp32(h: number): number {
  const s = (h & 0x8000) >> 15
  const e = (h & 0x7c00) >> 10
  const f = h & 0x03ff
  if (e === 0) {
    // Zero or subnormal: 2^-14 * (f / 1024)
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024)
  }
  if (e === 0x1f) {
    // Inf or NaN
    return f ? NaN : (s ? -Infinity : Infinity)
  }
  // Normal: 2^(e-15) * (1 + f/1024)
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024)
}
