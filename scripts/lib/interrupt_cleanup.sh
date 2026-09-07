#!/usr/bin/env bash
# Stop work launched by a run script when it is interrupted from the terminal.

_INTERRUPT_CLEANUP_ROOT_PID="$$"

_interrupt_cleanup_descendants() {
  local parent_pid="$1" child_pid
  local children_file="/proc/${parent_pid}/task/${parent_pid}/children"
  local -a child_pids=()

  [[ -r "$children_file" ]] || return 0
  read -r -a child_pids < "$children_file" || return 0
  for child_pid in "${child_pids[@]}"; do
    _interrupt_cleanup_descendants "$child_pid"
    # SIGINT gives training code an opportunity to close cleanly. A process
    # may already have exited because the terminal sent it SIGINT as well.
    kill -INT "$child_pid" 2>/dev/null || true
  done
}

_interrupt_cleanup() {
  local signal="$1" exit_code="$2"

  # Do not re-enter while signalling the process tree.
  trap - INT TERM
  printf '\nReceived %s; stopping all child processes...\n' "$signal" >&2
  _interrupt_cleanup_descendants "$_INTERRUPT_CLEANUP_ROOT_PID"
  exit "$exit_code"
}

trap '_interrupt_cleanup INT 130' INT
trap '_interrupt_cleanup TERM 143' TERM
