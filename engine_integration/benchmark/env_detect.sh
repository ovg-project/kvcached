#!/bin/bash

is_in_docker() {
    if [ -f "/.dockerenv" ] || grep -qaE '(docker|containerd|kubepods)' /proc/1/cgroup; then
        return 0
    else
        return 1
    fi
}

if is_in_docker; then
    export IN_DOCKER=true
else
    export IN_DOCKER=false
fi