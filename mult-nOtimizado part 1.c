/**
 * @file main-multi.c
 * @brief Matrix operations: loading, saving, printing, and multiplication.
 * @details
 * This file defines a lightweight matrix structure (`tpMatrix`) and provides
 * basic operations such as binary I/O, formatted printing, and matrix
 * multiplication with OpenMP parallelization.
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <omp.h>
#include <getopt.h>
#include <assert.h>
#include <stdint.h>
#include <papi.h>
#include <openssl/evp.h>
#include <sys/stat.h>
#define STRING_SIZE 64*1024             /**< Maximum buffer size for string manipulation (64 KiB). */
#ifdef __OPTIMIZE__                     /**< Default output file name for benchmark result storage.*/
#define PROJECT_NAME "MAT-MULT-TN712-OPTIMIZE"    
#else
#define PROJECT_NAME "MAT-MULT-TN712-NON-OPTIMIZE"    
#endif
#define PAPAI_EVENTS_NUMBER 4
/**
 * @struct tpMatrix
 * @brief Represents a two-dimensional matrix of double-precision values.
 *
 * @var tpMatrix::m
 * Number of rows in the matrix.
 * @var tpMatrix::n
 * Number of columns in the matrix.
 * @var tpMatrix::v
 * Pointer to the matrix elements stored in **row-major order**.
 */
typedef struct {
    uint32_t m, n;  /**< Number of rows and columns. */
    double *v;      /**< Pointer to matrix values (row-major order). */
} tpMatrix;

double get_wall_time(void);
void print_matrix(const tpMatrix *A);
void load_binary(tpMatrix *A, char * filename);
void save_binary(tpMatrix *A, char * filename);
void matrix_multi(long long **p_values,
                  tpMatrix *  C,
                  const tpMatrix *  A,
                  const tpMatrix *  B);

uint32_t md5_from_memory(uint8_t *md_value, const unsigned char *data, size_t len);
void save_anwser(const int threads, 
                 const double elapsedtime, 
                 const uint64_t mem, 
                 const long long **p_values, 
                 const uint8_t *md5, 
                 const uint32_t md_len);
void help(void);

int main (int ac, char **av){

    double elapsedtime = 0.0;
    uint64_t mem = 0;
    int show_matrix = 0;
    int option_index = 0;
    int threads =  omp_get_num_procs();
    int input_opt = 0;
    tpMatrix     A,
                 B,
                 C;

    char filename_matrix_A[STRING_SIZE],
         filename_matrix_B[STRING_SIZE],
         filename_matrix_C[STRING_SIZE];

    filename_matrix_A[0] = 0;
    filename_matrix_B[0] = 0;
    filename_matrix_C[0] = 0;
    
    
    long long *p_values[PAPAI_EVENTS_NUMBER];
		
    if (ac == 1) help();
    struct option long_options[] =
    {
        {"help",     no_argument,  0, 'h'},
        {"answer-matrix-c",   optional_argument, 0, 'c'},
        {"matrix-a",   required_argument, 0, 'a'},
        {"matrix-b",   required_argument, 0, 'b'},
        {"threads",   optional_argument, 0, 't'},
        {"show",   no_argument, 0, 's'},
        
        {0, 0, 0, 0}
    };

    while ((input_opt = getopt_long (ac, av, "hc:a:b:t:s", long_options, &option_index)) != EOF){
        switch (input_opt)
        {
            case 'h':help();break;
            case 'c': strcpy(filename_matrix_C, optarg); break;
            case 'a': strcpy(filename_matrix_A, optarg); break;
            case 'b': strcpy(filename_matrix_B, optarg); break;
            case 't': threads = atoi(optarg);break;
            case 's': show_matrix = 1;break;
            
            default: help();break;
        }
    };
    assert(filename_matrix_A[0] != 0);
    assert(filename_matrix_B[0] != 0);
    

    printf("Matrix multiplication\n\n");
    printf(" - Matrix A: [%s]\n", filename_matrix_A);
    printf(" - Matrix B: [%s]\n", filename_matrix_B);
    
    
    if (filename_matrix_C[0] != 0)
        printf(" - Matrix C: [%s]\n", filename_matrix_C);

    load_binary(&A, filename_matrix_A);
    load_binary(&B, filename_matrix_B);
    C.m = A.m;
    C.n = B.n;
    C.v = (double*) malloc(sizeof(double) * C.m * C.n);
    mem = sizeof(double) * ( (A.m * A.n) + (B.m * B.n) + (C.m * C.n) );
    memset(C.v, 0x00,  C.m *  C.n * sizeof(double));



    omp_set_num_threads(threads);
    printf("\t - Threads used: %d\n", threads);
    printf("\t -  Memory used: %lu bytes\n", mem);
    for (uint64_t i = 0; i < PAPAI_EVENTS_NUMBER; i++){
        p_values[i] = (long long *) malloc(sizeof(long long) * threads);
        bzero(p_values[i], sizeof(long long) * threads);

        assert(p_values[i] != NULL);
    }
        

    assert(PAPI_library_init(PAPI_VER_CURRENT) == PAPI_VER_CURRENT);
    assert(PAPI_thread_init(&pthread_self) == PAPI_OK);


    //PAPI_start(p_event_set);
    elapsedtime = get_wall_time();
    matrix_multi(p_values, &C, &A, &B);
    elapsedtime = get_wall_time() - elapsedtime;

    if (show_matrix){
        printf("\n-------------------------------------------------------------------------------\n");
        print_matrix(&A);
        printf("\n-------------------------------------------------------------------------------\n");
        print_matrix(&B);
        printf("\n-------------------------------------------------------------------------------\n");
        print_matrix(&C);

    }
    
    

    //Save the answer based on the MD5 hash of matrix C.
    unsigned char md5_value[EVP_MAX_MD_SIZE];
    uint32_t len = md5_from_memory(&md5_value[0], (const unsigned char*)C.v,  ( C.m * C.n * sizeof(double)));
    save_anwser(threads, elapsedtime, mem, (const long long **)p_values, &md5_value[0], len);

    for (uint64_t i = 0; i < PAPAI_EVENTS_NUMBER; i++){
        free(p_values[i]);
    }

    free(A.v);
    free(B.v);
    free(C.v);

    return EXIT_SUCCESS;
}

/**
 * @brief Returns the current wall-clock time in seconds.
 *
 * @details
 * This function retrieves the system’s monotonic clock time using
 * `clock_gettime()` with the `CLOCK_MONOTONIC` source, which measures
 * the time elapsed since the last system boot.
 *
 * The monotonic clock is not affected by system clock adjustments
 * (e.g., NTP corrections or manual time changes), making it suitable
 * for benchmarking and performance measurements.
 *
 * The result is returned as a double-precision floating-point value
 * in seconds, combining both seconds and nanoseconds from the
 * `timespec` structure.
 *
 * **Example usage:**
 * ```c
 * double start = get_wall_time();
 * run_simulation();
 * double elapsed = get_wall_time() - start;
 * printf("Elapsed time: %.6f s\n", elapsed);
 * ```
 *
 * @return The current wall-clock time in seconds as a double.
 * @retval 0.0 if an error occurs when calling `clock_gettime()`.
 *
 * @note
 * **Do not modify this function.**
 * It is used by the automatic performance evaluation system to measure
 * execution time precisely and consistently across different runs.
 *
 * @warning
 * - Requires POSIX support for `clock_gettime()`.
 * - Always returns a non-decreasing value (monotonic).
 */

double get_wall_time(void) {
    struct timespec ts;
    // CLOCK_MONOTONIC mede o tempo desde o último boot do sistema,
    // garantindo que não regride.
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == -1) {
        perror("clock_gettime");
        return 0.0;
    }
    // Converte segundos e nanossegundos para um valor double em segundos
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}
/**
 * @brief Prints a matrix to the standard output.
 *
 * @details
 * Displays the dimensions and elements of the matrix `A` using
 * a formatted, human-readable layout. Each element is printed with
 * eight decimal digits of precision and separated by semicolons.
 *
 * @param[in] A Pointer to the matrix to print.
 *
 * @note
 * Intended primarily for debugging or verification purposes.
 */
void print_matrix(const tpMatrix *A){
    printf("\t Print matrix (%u, %u)\n", A->m, A->n);
    for (uint32_t j = 0; j < A->m; j++){
        for (uint32_t i = 0; i < A->m; i++){
            printf("% 15.8lf;", A->v[j * A->n + i]);
        }//for (uint32_t i = 0; i < A->m; i++){
        printf("\n");
    }//for (uint32_t i = 0; i < A->m; i++){

}
/**
 * @brief Loads a matrix from a binary file.
 *
 * @details
 * Reads the matrix dimensions (`m`, `n`) and data elements from a binary file.
 * The file is expected to contain two 32-bit unsigned integers followed by
 * `m × n` double-precision floating-point values in row-major order.
 *
 * **Binary file layout:**
 * ```
 * [uint32_t m][uint32_t n][double v[m*n]]
 * ```
 *
 * @param[out] A Pointer to the destination matrix structure.
 * @param[in] filename Path to the binary file to read.
 *
 * @note
 * Allocates memory dynamically for `A->v`. The caller is responsible for
 * freeing it when no longer needed.
 *
 * @warning
 * The function asserts if the file cannot be opened.
 */ 
void load_binary(tpMatrix *A, char * filename){
    FILE *input = fopen(filename, "rb");
    uint64_t bytesRead = 0;
    
    assert(input != NULL);

    bytesRead = fread(&A->m, sizeof(uint32_t), 1, input);
    bytesRead += fread(&A->n, sizeof(uint32_t), 1, input);
    bytesRead *= sizeof(uint32_t);

    
    A->v = (double*) malloc(sizeof(double) * A->m * A->n);
    bytesRead += fread(A->v, sizeof(double), A->m * A->n, input) * sizeof(double);
    printf("\t load_binary - bytes read [%lu]\n", bytesRead);
    fclose(input);
     

}


/**
 * @brief Saves a matrix to a binary file.
 *
 * @details
 * Writes the matrix dimensions (`m`, `n`) followed by the data array `v`
 * to a binary file in row-major order. The file will be overwritten if it exists.
 *
 * **Binary file layout:**
 * ```
 * [uint32_t m][uint32_t n][double v[m*n]]
 * ```
 *
 * @param[in] A Pointer to the matrix structure to save.
 * @param[in] filename Path to the output binary file.
 *
 * @warning
 * The function asserts if the file cannot be opened for writing.
 */
void save_binary(tpMatrix *A, char * filename){
    FILE *output = fopen(filename, "wb+");
    double *v  = A->v;
    uint64_t bytesWrite = 0;
    assert(output != NULL);

    bytesWrite = fwrite(&A->m, sizeof(uint32_t), 1, output);
    bytesWrite += fwrite(&A->n, sizeof(uint32_t), 1, output);
    bytesWrite *= sizeof(uint32_t);
    bytesWrite += fwrite(v, sizeof(double), A->m * A->n, output) * sizeof(double);
    printf("\t save_binary - bytes written [%lu]\n", bytesWrite);
    fclose(output);

}

/**
 * @brief Performs matrix multiplication using OpenMP parallelization.
 *
 * @details
 * Computes the product of matrices `A` and `B`, storing the result in `C`.
 * All matrices are assumed to be stored in **row-major order**, and their
 * dimensions must satisfy:
 * \f[
 *     A_{m \times n} \times B_{n \times p} = C_{m \times p}
 * \f]
 *
 * Parallelism is applied to the outer loop over matrix rows (`j`) using OpenMP.
 *
 * @param[out] C Pointer to the result matrix.
 * @param[in]  A Pointer to the left-hand operand matrix.
 * @param[in]  B Pointer to the right-hand operand matrix.
 *
 * @note
 * The function assumes that `C->v` has been pre-allocated with
 * at least `C->m * C->n` elements.
 *
 * @warning
 * No dimension compatibility check is performed at runtime.
 */
void matrix_multi(long long **p_values,
                  tpMatrix *  C,
                  const tpMatrix *  A,
                  const tpMatrix *  B){

    #pragma omp parallel shared(C, A, B, p_values)
    {
        int EventSet = PAPI_NULL;
        long long values_local[PAPAI_EVENTS_NUMBER];

        PAPI_register_thread();
        assert(PAPI_create_eventset(&EventSet) == PAPI_OK);

        assert(PAPI_add_event(EventSet, PAPI_L1_DCM) == PAPI_OK);
        assert(PAPI_add_event(EventSet, PAPI_L2_DCM) == PAPI_OK);
        assert(PAPI_add_event(EventSet, PAPI_FP_INS) == PAPI_OK);
        assert(PAPI_add_event(EventSet, PAPI_MEM_SCY) == PAPI_OK);

        assert(PAPI_start(EventSet) == PAPI_OK);

        #pragma omp for  
        for (uint32_t j = 0; j < C->m; j++){
            for (uint32_t i = 0; i < C->n; i++){
                double c = 0.0f;
                for (uint32_t jA = 0; jA < A->n; jA++){
                    uint32_t ak = j * A->n + jA;
                    uint32_t bk = jA * B->n + i;
                    c += A->v[ak] * B->v[bk];
                }//for (int jA = j; jA < A->m; jA++){

                uint32_t ck = j * C->n +  i;
                C->v[ck] = c;
            }//end-for (int j = 0; j < mLattice->height; j++){

        }//end-void InitRandness(tpLattice *mLattice, float p){
        assert(PAPI_stop(EventSet, values_local) == PAPI_OK);
        for (uint64_t i = 0; i < PAPAI_EVENTS_NUMBER; i++){
            p_values[i][omp_get_thread_num()] = values_local[i];
        }
        
        assert(PAPI_cleanup_eventset(EventSet) == PAPI_OK);;
        assert(PAPI_destroy_eventset(&EventSet) == PAPI_OK);;
        PAPI_unregister_thread();
    }//#pragma omp parallel shared(C, A, B)
    PAPI_shutdown();
}


/**
 * @brief Computes the MD5 hash of a memory buffer.
 *
 * @details
 * This function generates an MD5 digest from a given memory block using
 * the OpenSSL EVP (Envelope) interface. The resulting hash is stored
 * in the buffer pointed to by `md_value`, and the function returns
 * the digest length in bytes.
 *
 * Internally, the function initializes an MD5 context, processes the input
 * data with `EVP_DigestUpdate()`, finalizes the digest with
 * `EVP_DigestFinal_ex()`, and then frees the context.
 *
 * **Example usage:**
 * ```c
 * uint8_t md5_hash[EVP_MAX_MD_SIZE];
 * uint32_t md_len = md5_from_memory(md5_hash, buffer, buffer_length);
 * ```
 *
 * **Example output:**
 * ```
 * MD5 digest (hex): 9e107d9d372bb6826bd81d3542a419d6
 * ```
 *
 * @param[out] md_value Pointer to the output buffer where the MD5 digest will be stored.
 * @param[in]  data     Pointer to the input data to be hashed.
 * @param[in]  len      Length of the input data in bytes.
 *
 * @return The length (in bytes) of the computed MD5 digest.
 *
 * @note
 * **Do not modify this function.**
 * It is used by the automated validation system to verify simulation results
 * and ensure data integrity. Changing its behavior will invalidate checksums.
 *
 * @warning
 * - The output buffer `md_value` must be large enough to hold the digest
 *   (at least `EVP_MAX_MD_SIZE` bytes).
 * - Requires OpenSSL's EVP interface.
 */
uint32_t md5_from_memory(uint8_t *md_value, const unsigned char *data, size_t len) {
    //unsigned char md_value[EVP_MAX_MD_SIZE];
    uint32_t md_len;
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();

    EVP_DigestInit_ex(ctx, EVP_md5(), NULL);
    EVP_DigestUpdate(ctx, data, len);
    EVP_DigestFinal_ex(ctx, md_value, &md_len);
    EVP_MD_CTX_free(ctx);
    return md_len;
    //printf("\nMD5: ");
    //for (unsigned int i = 0; i < md_len; i++)
    //    printf("%02x", md_value[i]);
    //printf("\n");
}

/**
 * @brief Saves the benchmark answer file containing the MD5 checksum.
 *
 * @details
 * This function writes the simulation's performance result and verification
 * hash to a file named after the project (defined by `PROJECT_NAME`).
 * The file contains a single line with the MD5 digest in hexadecimal format.
 *
 * **Example output:**
 * ```
 *     [a1b2c3d4e5f67890abcdef1234567890]
 * ```
 *
 
 * @param[in] md5          Pointer to the MD5 digest array.
 * @param[in] md_len       Length of the MD5 digest (number of bytes).
 *
 * @note
 * **Do not modify this function.**
 * It is part of the automatic evaluation system and is used to validate
 * the program’s output. Changing it may invalidate the results.
 *
 * @warning
 * The file is opened in write mode ("w+"), overwriting any existing content
 * associated with the same `PROJECT_NAME` file.
 */
void save_anwser(const int threads, 
                 const double elapsedtime, 
                 const uint64_t mem, 
                 const long long **p_values, 
                 const uint8_t *md5, 
                 const uint32_t md_len){
    char file_name[STRING_SIZE];
    struct stat st;
    sprintf(file_name, "%s-%03d.csv", PROJECT_NAME, threads);
    printf("\t - Saving [%s]", file_name);
    FILE *ptr = NULL;

    
    if (!stat(file_name, &st) == 0) {
        ptr = fopen(file_name, "w+");
        assert(ptr != NULL);
        fprintf(ptr, "threads;elapsedtime;mem;md5_anwser");
        for (int i = 0; i < threads; i++){
            fprintf(ptr,
            ";t-%d-L1_DCM"
            ";t-%d-L2_DCM"
            ";t-%d-FP_INS"
            ";t-%d-MEM_SCY",
            i,i,i,i
);

        }
        fprintf(ptr, "\n");
    } else {
        ptr = fopen(file_name, "a+");
        assert(ptr != NULL);
    }

    fprintf(ptr, "%d;%lf;%u;", threads, elapsedtime, mem);
    for (uint8_t i = 0; i < md_len; i++)fprintf(ptr, "%02x", md5[i]);
    for (int i = 0; i < threads; i++)
    {
        for (int e = 0; e < PAPAI_EVENTS_NUMBER; e++)
            fprintf(ptr, ";%lld", p_values[e][i]);
    }

     fprintf(ptr, "\n");

  
    fclose(ptr);

    printf("\t OK\n");
}

/**
 * @brief Display help and usage instructions.
 * @details Prints command-line argument options and usage information, then exits.
 */
void help(void){
    fprintf(stdout, "\nMatrix multiplication\n");
    fprintf(stdout, "Usage: ../m_mult.exec [ -c matrix C.bin ] < -a matrix A.bin > < -b matrix B.bin > [ -t threads ] [ -s ]\n");
    fprintf(stdout, "**Options:**\n");
    fprintf(stdout, "\t'-h', '--help': Show this help message\n");
    
    fprintf(stdout, "\t'-c', '--answer-matrix-c': Optional parameter. It represent the name of the result matrix, denoted as C.\n");
    fprintf(stdout, "\t'-a', '--matrix-a': Matrix A.\n");
    fprintf(stdout, "\t'-b', '--matrix-b': Matrix B.\n");
    fprintf(stdout, "\t'-t', '--threads': Number of threads allocated.\n");
    fprintf(stdout, "\t'--show': It prints the matrices (A), (B) and (C).\n");
    
    exit(EXIT_FAILURE);
}
