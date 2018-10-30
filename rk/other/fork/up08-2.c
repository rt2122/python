#include <stdio.h>
#include <stdlib.h>
#include <wait.h>
#include <unistd.h>

int main()
{
    int pid = fork();
    if (!pid) {

        int pid1 = fork();
        if (!pid1) {
            printf("3 ");
            exit(0);
        } else {
            printf("2 ");
            wait(NULL);
            exit(0);
        }
    } else {
        wait(NULL);
        printf("1\n");
    }

    return 0;
}
