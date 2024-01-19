#include <stdio.h>
#include "structures.h"
#include "network.c"
#include "read_csv.c"

int main() {
    input_tensor *n = (input_tensor *)malloc(1 * sizeof(input_tensor));

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            n->w1[i][j] = getRandomNumber();
        }
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            n->w2[i][j] = getRandomNumber();
        }
    }

    for (int epoch = 0; epoch < 10; epoch++) {
        printf("Epoch %d\n", epoch + 1);
        float loss = 0;
        int correct_predictions = 0;

        for (int i = 0; i < 10000; i++) {
            image *data = input_array(i + 1);
            memcpy(n->value, data->value, INPUT_SIZE * sizeof(float));
            float *encoded_label = one_hot_encode(data->label, 10);
            memcpy(n->label, encoded_label, OUTPUT_SIZE * sizeof(float));
            free(encoded_label);
            free(data);

            loss += categorical_cross_entropy_loss(n->a2, n->label);

            int predicted_class = 0;
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (n->a2[j] > n->a2[predicted_class]) {
                    predicted_class = j;
                }
            }

            if (predicted_class == data->label) {
                correct_predictions++;
            }

            backward(n);
        }

        float accuracy = (float)correct_predictions / 10000.0 * 100.0;
        printf("Loss: %.2f, Accuracy: %.2f%%\n", loss, accuracy);
    }

    free(n);
    return 0;
}
