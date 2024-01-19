#include <math.h>

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

void softmax(float *x, float *result, int n) {
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += exp(x[i]);
    }
    for (int i = 0; i < n; i++) {
        result[i] = exp(x[i]) / sum;
    }
}

float ReLu(float x) {
    if (x > 0)
        return x;
    else
        return 0;
}

void relu_backward(const float *dA, const float *Z, float *dZ, int size) {
    for (int i = 0; i < size; i++) {
        dZ[i] = (Z[i] > 0) ? dA[i] : 0;
    }
}

void softmax_backward(const float *dA, const float *Z, float *dZ, int size) {
    for (int i = 0; i < size; i++) {
        float sig = sigmoid(Z[i]);
        dZ[i] = dA[i] * sig * (1.0 - sig);
    }
}


