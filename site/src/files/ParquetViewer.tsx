// Use the lib's reference parquet renderer instead of carrying our own copy.
// Available via `@rdub/file-tree/renderers/parquet` since file-tree commit
// 8005f43 (spec `export-reference-renderers.md`). hyparquet is an optional
// peer dep there — we already have it as a transitive dep via the runs
// dashboard's parquet reader.
export { ParquetViewer } from '@rdub/file-tree/renderers/parquet'
