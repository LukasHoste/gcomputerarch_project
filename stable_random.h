#ifndef STABLE_RANDOM_H
#define STABLE_RANDOM_H
typedef struct {
    unsigned long long seed;
} StableRandom;

// Constants for 64-bit LCG (carefully chosen)
#define MULTIPLIER 6364136223846793005ULL
#define INCREMENT 1ULL
#define MODULUS (1ULL << 64)

void stablerand_init(StableRandom *rng, unsigned long long seed) {
    rng->seed = seed;
}

// Generate next random number between 0 and 1
double stablerand_next(StableRandom *rng) {
    rng->seed = (rng->seed * MULTIPLIER + INCREMENT); // Modulus 2^64 is automatic for unsigned long long
    return (rng->seed >> 11) * (1.0 / (1ULL << 53)); // Scale down to [0,1)
}
#endif // STABLE_RANDOM_H