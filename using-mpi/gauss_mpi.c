/* Gaussian elimination without pivoting.
 * Compile with "gcc gauss.c"
 */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>
#include <string.h>

/* Program Parameters */
#define MAXN 2000  /* Max value of N */
int N; /* Matrix size */

/* Name given to 0th process */
#define MASTER 0

/* Processes size */
int totalProcesses;

/* Process rank */
int myProcessRank;

/* Output file name*/
char *outputFileName;

/* Matrices and vectors */
float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss(); /* The function you will provide.
 * It is this routine that is timed.
 * It is called only on the parent.
 */

/* Back Substitution Method Prototype */
void backSubstitution();

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int) (t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  int seed = 0; /* Random seed */
  char uid[32]; /*User name */
  /* Read command-line arguments */
  srand(time_seed()); /* Randomize */

  if (argc == 4) {

    seed = atoi(argv[2]);
    srand(seed);

    if (myProcessRank == MASTER) {
      printf("\nGauss Elimination using mpi\n");
      printf("\nRandom seed = %i\n", seed);
      int length = strlen(argv[3]);
      outputFileName = (char*) malloc(length + 1);
      outputFileName = argv[3];
      printf("\nOutput file = %s\n", outputFileName);

    }

    N = atoi(argv[1]);

    if (N < 1 || N > MAXN) {
      if (myProcessRank == MASTER) {
        printf("N = %i is out of range.\n", N);
      }
      exit(0);
    }
  } else {
    if (myProcessRank == MASTER) {
      printf("Usage: %s <matrix_dimension> <random seed> <output_file_name>\n", argv[0]);
    }
    exit(0);
  }

  /* Print parameters */
  if (myProcessRank == MASTER) {
    printf("\nMatrix dimension N = %i.\n", N);
  }
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float) rand() / 32768.0;
    }
    B[col] = (float) rand() / 32768.0;
    X[col] = 0.0;
  }

}

/* Print input matrices */
void print_inputs(FILE *filePtr) {
  int row, col;

  if (N < 10) {
    fprintf(filePtr, "\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
        fprintf(filePtr, "%9.6f%s", A[row][col], (col < N - 1) ? ", " : ";\n\t");
      }
    }
    fprintf(filePtr, "\nB = [");
    for (col = 0; col < N; col++) {
      fprintf(filePtr, "%9.6f%s", B[col], (col < N - 1) ? "; " : "]\n");
    }
  }
}

void print_X(FILE *filePtr) {
  int row;

  if (N < 100) {
    fprintf(filePtr, "\nX = [");
    for (row = 0; row < N; row++) {
      fprintf(filePtr, "%9.6f%s", X[row], (row < N - 1) ? "; " : "]\n");
    }
  }
}

int main(int argc, char **argv) {
  /* Timing variables */
  struct timeval etstart, etstop; /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  clock_t etstart2, etstop2; /* Elapsed times using times() */
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop; /* CPU times for my processes */
  FILE *filePtr;

  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  // Get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);

  // Get process rank
  MPI_Comm_rank(MPI_COMM_WORLD, &myProcessRank);

  /* Process program parameters */
  parameters(argc, argv);

  /* Open file to write the results */
  if (myProcessRank == MASTER) {
    filePtr = fopen(outputFileName, "w+");
    if (filePtr == NULL) {
      printf("unable to create/override the file");
    }

    /* Initialize A and B */
    initialize_inputs();

    /* Print input matrices */
    print_inputs(filePtr);
    printf("\nStarting clock.\n");
    /* Start Clock */
    gettimeofday(&etstart, &tzdummy);
    etstart2 = times(&cputstart);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  /* Gaussian Elimination */
  gauss();

  if (myProcessRank == MASTER) {

    print_inputs(filePtr);

    /* Stop Clock */
    gettimeofday(&etstop, &tzdummy);
    etstop2 = times(&cputstop);
    printf("\nStopped clock.\n");
    usecstart = (unsigned long long) etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long) etstop.tv_sec * 1000000 + etstop.tv_usec;

    /* Display output */
    print_X(filePtr);

    /* Display timing results */
    fprintf(filePtr, "\nElapsed time = %g ms.\n", (float) (usecstop - usecstart) / (float) 1000);

    fprintf(filePtr, "(CPU times are accurate to the nearest %g ms)\n", 1.0 / (float) CLOCKS_PER_SEC * 1000.0);
    fprintf(filePtr, "My total CPU time for parent = %g ms.\n", (float) ((cputstop.tms_utime + cputstop.tms_stime) - (cputstart.tms_utime + cputstart.tms_stime)) / (float) CLOCKS_PER_SEC * 1000);
    fprintf(filePtr, "My system CPU time for parent = %g ms.\n", (float) (cputstop.tms_stime - cputstart.tms_stime) / (float) CLOCKS_PER_SEC * 1000);
    fprintf(filePtr, "My total CPU time for child processes = %g ms.\n",
        (float) ((cputstop.tms_cutime + cputstop.tms_cstime) - (cputstart.tms_cutime + cputstart.tms_cstime)) / (float) CLOCKS_PER_SEC * 1000);
    /* Contrary to the man pages, this appears not to include the parent */
    fprintf(filePtr, "--------------------------------------------\n");
    printf("\nResults are generated into %s file\n", outputFileName);

  }

  MPI_Finalize();

  exit(0);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */
void gauss() {

  int norm, row, col, multiplier[N], rownum[N];

  /* Master Process (0) broadcast the 0th row to all processes*/
  MPI_Bcast(&A[0][0], MAXN * MAXN, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(B, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  for (row = 0; row < N; row++) {
    rownum[row] = row % totalProcesses;
  }

  for (norm = 0; norm < N; norm++) {
    MPI_Bcast(&A[norm][norm], N - norm, MPI_FLOAT, rownum[norm], MPI_COMM_WORLD);
    MPI_Bcast(&B[norm], 1, MPI_FLOAT, rownum[norm], MPI_COMM_WORLD);
    for (row = norm + 1; row < N; row++) {
      if (rownum[row] == myProcessRank) {
        multiplier[row] = A[row][norm] / A[norm][norm];
      }
    }
    for (row = norm + 1; row < N; row++) {
      if (rownum[row] == myProcessRank) {
        for (col = 0; col < N; col++) {
          A[row][col] = A[row][col] - (multiplier[row] * A[norm][col]);
        }
        B[row] = B[row] - (multiplier[row] * B[norm]);
      }
    }
  }

  /* (Diagonal elements are not normalized to 1.  This is treated in back
   * substitution.)
   */
  backSubstitution();

}

/* Back substitution */
void backSubstitution() {
  int row, col;
  if (myProcessRank == MASTER) {
    for (row = N - 1; row >= 0; row--) {
      X[row] = B[row];
      for (col = N - 1; col > row; col--) {
        X[row] -= A[row][col] * X[col];
      }
      X[row] /= A[row][row];
    }

  }
}
