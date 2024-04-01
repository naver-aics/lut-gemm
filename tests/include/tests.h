#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <memory>

/* submodule */
#include "gtest/gtest.h"

/* custom module */
#include "custom_random.h"
#include "timer.h"

#include <sys/time.h>

#include "lutGEMM"

#ifndef GTEST_PIRNTF
#define GTEST_PIRNTF(...){\
    printf("\033[32m[          ]");\
    printf("\033[0m ");\
    printf(__VA_ARGS__);\
    printf("\n");\
}
#endif
