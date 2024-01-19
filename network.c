#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "activation.c"
#include "structures.h"

float getRandomNumber() {
    return (float)rand() / RAND_MAX * 0.01 - 0.005;
}

float* forward(input_tensor *n) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        n->z1[i] = 0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            n->z1[i] += n->value[j] * n->w1[j][i];
        }
        n->a1[i] = ReLu(n->z1[i]);
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        n->z2[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            n->z2[i] += n->a1[j] * n->w2[j][i];
        }
    }

    softmax(n->z2, n->a2, OUTPUT_SIZE);

    return n->a2;
}

float* one_hot_encode(int label, int num_classes) {
    float *encoded = (float *)malloc(num_classes * sizeof(float));
    for (int i = 0; i < num_classes; i++) {
        encoded[i] = 0.0;
    }
    encoded[label] = 1.0;

    return encoded;
}

float categorical_cross_entropy_loss(float* predictions, float* target_labels) {
    float loss = 0.0f;

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        loss += -target_labels[i] * log(predictions[i] + EPSILON);
    }

    return loss;
}

void gradient_descent_update(input_tensor *n, float learning_rate) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            n->w2[j][i] -= learning_rate * n->dA2[i] * n->a1[j];
        }
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            n->w1[j][i] -= learning_rate * n->dA1[i] * n->value[j];
        }
    }
}

void backward(input_tensor *n) {
    float *output = forward(n);

    float dA2[OUTPUT_SIZE];
    softmax_backward(output, n->label, dA2, OUTPUT_SIZE);
    memcpy(n->dA2, dA2, OUTPUT_SIZE * sizeof(float));

    gradient_descent_update(n, LEARNING_RATE);

    float dZ2[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        dZ2[i] = 0;
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            dZ2[i] += dA2[k] * n->w2[i][k];
        }
    }

    relu_backward(dZ2, n->a1, n->z1, HIDDEN_SIZE);
    memcpy(n->dA1, dZ2, HIDDEN_SIZE * sizeof(float));
}
