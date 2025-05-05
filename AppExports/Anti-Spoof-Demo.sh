#!/bin/sh
echo -ne '\033c\033]0;Anti-Spoof-Demo\a'
base_path="$(dirname "$(realpath "$0")")"
"$base_path/Anti-Spoof-Demo.x86_64" "$@"
