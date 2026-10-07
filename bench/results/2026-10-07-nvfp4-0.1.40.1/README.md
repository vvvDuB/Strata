# NVFP4 and upstream 0.1.40.1 validation

See [validation.json](validation.json) for the measured hardware, test counts,
real model trials and unavailable gates. Integration details and reproduction
commands are in [UPSTREAM_0_1_40_1.md](../../../docs/UPSTREAM_0_1_40_1.md).

The tests distinguish automated successes, owner-supplied privileged mlock
validation, explicit hardware skips and missing independent PLE fixtures.
The executable checksum identifies the native CUDA 13.4 / SM120 release build.
