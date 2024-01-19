#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "structures.h"

image* input_array(int line_number) {
    FILE *file = fopen("mnist_train.txt", "r");

    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    char line[3000];
    int current_line = 0;

    while (current_line < line_number && fgets(line, sizeof(line), file) != NULL) {
        current_line++;
    }

    if (current_line < line_number) {
        printf("Line %d not found. File may be too short.\n", line_number);
        fclose(file);
        return NULL;
    }

    const char *delimiter = ",";
    char *token = strtok(line, delimiter);
    int index = 0;
    image *data = (image*)malloc(sizeof(image));

    char *endptr;
    data->label = strtof(token, &endptr);
    if (*endptr != '\0' && *endptr != '\n') {
        fprintf(stderr, "Error converting token to float: %s\n", token);
        fclose(file);
        free(data);
        return NULL;
    }

    token = strtok(NULL, delimiter);
    while (token != NULL && index < 784) {
        data->value[index++] = strtof(token, &endptr);
        if (*endptr != '\0' && *endptr != '\n') {
            fprintf(stderr, "Error converting token to float: %s\n", token);
            fclose(file);
            free(data);
            return NULL;
        }
        token = strtok(NULL, delimiter);
    }

    fclose(file);
    return data;
}