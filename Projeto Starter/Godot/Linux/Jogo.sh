#!/bin/sh
echo -ne '\033c\033]0;ILTutorialEnv\a'
base_path="$(dirname "$(realpath "$0")")"
"$base_path/Jogo.x86_64" "$@"
