. $(dirname "$0")/ci-common.sh

export RUST_BACKTRACE=1
# Run all tests with 1G heap
export MMTK_GC_TRIGGER=FixedHeapSize:1000000000

for_all_features "cargo test"

# target-specific features
if [[ $arch == "x86_64" && $os == "linux" ]]; then
    cargo test --features perf_counter
fi

ALL_PLANS=$(sed -n '/enum PlanSelector/,/}/p' src/util/options.rs | sed -e 's;//.*;;g' -e '/^$/d' -e 's/,//g' | xargs | grep -o '{.*}' | grep -o '\w\+')

# At the moment, the Compressor does not work with the mock VM tests.
# So we skip testing the Compressor entirely.
ALL_PLANS=$(echo -n "$ALL_PLANS" | sed '/Compressor/d')
# LXR requires side mark bits (its Immix block code extracts the side spec);
# MockVM keeps the mark bit in the header, so LXR cannot run under it.
ALL_PLANS=$(echo -n "$ALL_PLANS" | sed '/LXR/d')

# The mock tests were written for the 32KB bump-allocator granule and a 1MB
# fixture heap; the fork's measured default is 512KB (see bumpallocator.rs),
# whose first block plus its copy reserve already exceeds that heap. Pin the
# granule for the mock tests only.
export MMTK_BUMP_BLOCK_KB=32

# Test with mock VM:
# - Find all the files that start with mock_test_
# - Run each file separately with cargo test, with the feature 'mock_test'
find ./src ./tests -type f -name "mock_test_*" | while read -r file; do
    t=$(basename -s .rs $file)

    # Get the required plans.
    # Some tests need to be run with multiple plans because
    # some bugs can only be reproduced in some plans but not others.
    PLANS=$(sed -n 's/^\/\/ *GITHUB-CI: *MMTK_PLAN=//p' $file | tr ',' '\n')
    if [[ $PLANS == 'all' ]]; then
        PLANS=$ALL_PLANS
    elif [[ -z $PLANS ]]; then
        PLANS=NoGC
    fi

    # Some tests need some features enabled.
    FEATURES=$(sed -n 's/^\/\/ *GITHUB-CI: *FEATURES=//p' $file)

    # Run the test with each plan it needs.
    for MMTK_PLAN in $PLANS; do
        env MMTK_PLAN=$MMTK_PLAN cargo test --features mock_test,"$FEATURES" -- $t;
    done
done

# Test the dummy VM
cargo test --manifest-path $dummyvm_toml
