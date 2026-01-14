#pragma once

static inline const char *alvr_build_stamp(void) {
    return __DATE__ " " __TIME__;
}
