# Security Model

## Shell Job Execution

Shell jobs in the Minions queue allow arbitrary command execution.
This is **disabled by default** and requires `KCT_ENABLE_SHELL_JOBS=1`.

**Warning**: Only enable in trusted local automation environments.
Never accept shell job payloads from external/untrusted input.
