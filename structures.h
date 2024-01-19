#ifndef STRUCTURES_H
#define STRUCTURES_H
#define INPUT_SIZE 784
#define HIDDEN_SIZE 512
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.001
#define EPSILON 1e-10

typedef struct {
    float label;
    float value[784];
} image;

typedef struct {
    float label[10];
    float value[784];
    float w1[INPUT_SIZE][HIDDEN_SIZE];
    float z1[HIDDEN_SIZE];
    float a1[HIDDEN_SIZE];
    float w2[HIDDEN_SIZE][OUTPUT_SIZE];
    float z2[OUTPUT_SIZE];
    float a2[OUTPUT_SIZE];
    float dA1[HIDDEN_SIZE];
    float dA2[OUTPUT_SIZE];
} input_tensor;

#endif
