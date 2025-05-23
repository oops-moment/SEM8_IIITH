#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
 
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv); 
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size != 2)
    {
        printf("This application is meant to be run with 2 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
 
    enum role_ranks { SENDER, RECEIVER };
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    switch(my_rank)
    {
        case SENDER:
        {
            int buffer_sent = 12345;
            int buffer_sent2 = 54321;
            printf("MPI process %d sends value %d.\n", my_rank, buffer_sent);
            MPI_Send(&buffer_sent, 1, MPI_INT, RECEIVER, 0, MPI_COMM_WORLD);
            MPI_Send(&buffer_sent2, 1, MPI_INT, RECEIVER, 2, MPI_COMM_WORLD);
            break;
        }
        case RECEIVER:
        {
            int received;
            MPI_Recv(&received, 1, MPI_INT, SENDER, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("MPI process %d received value: %d.\n", my_rank, received);
            break;
        }
    }
 
    MPI_Finalize(); 
    return EXIT_SUCCESS;
}

