# Legacy cache writeback: avoid the synchronous durability barrier

The Q3 45k watchdog abort previously described as a possible CUDA stall was
identified from its original Linux core dump. The serving thread was blocked in
`fsync -> prefix_write -> ConversationStore::put`, before the new prompt, with
the expert pool parked and the previous verify window complete. The watchdog
aborted after 60 seconds. This is disk writeback blocking inference, not evidence
of a GPU flag deadlock. A separate older 0.1.28 core stopped in scalar checksum
calculation; it must not be assigned the same cause without evidence.

## Fix and contract

Legacy `prefix_write` now uses OS writeback rather than calling `fsync` or
`fdatasync` from the serving thread. It still flushes userspace buffers, checks
flush **and close** errors, checks cancellation, then atomically links the private
temporary file without overwriting an existing image. The immutable checkpoint
loans remain synchronous and are not retained. No background publisher, queue,
extra thread, configuration option or watchdog timeout increase is introduced.

These files are derived inference caches, not authoritative conversation data.
They are immediately readable and survive an ordinary engine/process restart,
but are **not guaranteed durable across a power failure or kernel crash**. An
image can disappear or be incomplete; bounds, identity, record checksums and
footer validation must succeed before GPU restore, otherwise reuse falls back to
a cache miss. Client conversation/history storage is unchanged.

The optional newer shared-NVMe tier retains its existing durability semantics;
it is not enabled by the owner's Q3 profile and is outside this fix. Normal
`fwrite`/`fflush` can still be throttled by a severely overloaded filesystem.
This fix removes the observed explicit barrier, not every conceivable disk,
driver, kernel or GPU hang. The real-stall watchdog remains enabled at 60 seconds.

## Regression gates

`prefix_writeback_test` is registered for Linux in both the standalone and parent
CMake tests. It injects slow/failing barriers and verifies that none is called,
publication is immediately readable and private/no-overwrite, cancellation and
close errors never publish, and a truncated image is rejected. It was observed
RED on the original implementation, then GREEN; ASan/UBSan also cover this path.

An actual Q3 HTTP A/B/A test interposes a 70-second barrier on legacy temporary
files. The original installed V6 aborts at B via the unchanged 60-second watchdog,
matching the original core. The corrected build completes all three requests
and restores A from disk, with no barrier invoked. Longer Q3 repeat/edit/append
and branch-switch testing uses 12 balanced checkpoints at 4096, INT8 KV, mmap,
MTP and 128k **allocated** context; no full 128k input is implied.

Deployment logs and fault-injection artifacts are private; core dumps may contain
request/model data and credentials and must not be published with this document.
