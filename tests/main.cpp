#include <gtest/gtest.h>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Now argc/argv contain only *your* custom args.
    // Google Test removes its own.
    // for (int i = 1; i < argc; ++i)
    //    ...whatever...

    return RUN_ALL_TESTS();
}
