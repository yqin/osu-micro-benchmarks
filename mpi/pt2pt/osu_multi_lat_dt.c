#define BENCHMARK "OSU MPI Multi Latency Test"
/*
 * Copyright (C) 2002-2021 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include <osu_util_mpi.h>

char *s_buf, *r_buf;

static void multi_latency(int rank, int pairs);

int main(int argc, char* argv[])
{
    int rank, nprocs; 
    int pairs;
    int po_ret = 0;
    options.bench = PT2PT;
    options.subtype = LAT_DT;

    set_header(HEADER);
    set_benchmark_name("osu_multi_lat_dt");

    po_ret = process_options(argc, argv);

    /* Sanity check */
    if (options.dt_block_size > options.dt_stride_size ||
        options.dt_block_size > MAX_DT_BLOCK_SIZE ||
        options.dt_stride_size > MAX_DT_STRIDE_SIZE) {
        po_ret = PO_BAD_USAGE;
    }

    if (options.dt_block_size > options.min_message_size) {
        options.min_message_size = options.dt_block_size;
    }

    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }
    MPI_CHECK(MPI_Init(&argc, &argv));

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

    pairs = nprocs/2;

    if (0 == rank) {
        switch (po_ret) {
            case PO_CUDA_NOT_AVAIL:
                fprintf(stderr, "CUDA support not enabled.  Please recompile "
                        "benchmark with CUDA support.\n");
                break;
            case PO_OPENACC_NOT_AVAIL:
                fprintf(stderr, "OPENACC support not enabled.  Please "
                        "recompile benchmark with OPENACC support.\n");
                break;
            case PO_BAD_USAGE:
                print_bad_usage_message(rank);
                break;
            case PO_HELP_MESSAGE:
                print_help_message(rank);
                break;
            case PO_VERSION_MESSAGE:
                print_version_message(rank);
                MPI_CHECK(MPI_Finalize());
                break;
            case PO_OKAY:
                break;
        }
    }

    switch (po_ret) {
        case PO_CUDA_NOT_AVAIL:
        case PO_OPENACC_NOT_AVAIL:
        case PO_BAD_USAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_FAILURE);
        case PO_HELP_MESSAGE:
        case PO_VERSION_MESSAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_OKAY:
            break;
    }

    if (allocate_memory_pt2pt_mul(&s_buf, &r_buf, rank, pairs)) {
        /* Error allocating memory */
        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    if(rank == 0) {
        print_header(rank, LAT);
        fflush(stdout);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    multi_latency(rank, pairs);
    
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    MPI_CHECK(MPI_Finalize());

    free_memory_pt2pt_mul(s_buf, r_buf, rank, pairs);

    return EXIT_SUCCESS;
}

static void multi_latency(int rank, int pairs)
{
    int size, partner;
    int i;
    double t_start = 0.0, t_end = 0.0,
           latency = 0.0, total_lat = 0.0,
           avg_lat = 0.0;

    MPI_Request request;
    MPI_Status reqstat;

    int rep_count;
    MPI_Datatype type;


    for(size = options.min_message_size; size <= options.max_message_size; size  = (size ? size * 2 : 1)) {

        if(size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        } else {
            options.iterations = options.iterations;
            options.skip = options.skip;
        }

        /* Define DDT */
        rep_count = size / options.dt_block_size;
        MPI_CHECK(MPI_Type_vector(rep_count, options.dt_block_size, options.dt_stride_size, MPI_CHAR, &type));
        MPI_CHECK(MPI_Type_commit(&type));

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        if (rank < pairs) {
            partner = rank + pairs;

            for (i = 0; i < options.iterations + options.skip; i++) {

                if (i == options.skip) {
                    t_start = MPI_Wtime();
                    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
                }

                MPI_CHECK(MPI_Isend(s_buf, 1, type, partner, 1, MPI_COMM_WORLD, &request));
                MPI_CHECK(MPI_Wait(&request, &reqstat));
                MPI_CHECK(MPI_Irecv(r_buf, 1, type, partner, 1, MPI_COMM_WORLD, &request));
                MPI_CHECK(MPI_Wait(&request, &reqstat));
            }

            t_end = MPI_Wtime();

        } else {
            partner = rank - pairs;

            for (i = 0; i < options.iterations + options.skip; i++) {

                if (i == options.skip) {
                    t_start = MPI_Wtime();
                    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
                }

                MPI_CHECK(MPI_Irecv(r_buf, 1, type, partner, 1, MPI_COMM_WORLD, &request));
                MPI_CHECK(MPI_Wait(&request, &reqstat));
                MPI_CHECK(MPI_Isend(s_buf, 1, type, partner, 1, MPI_COMM_WORLD, &request));
                MPI_CHECK(MPI_Wait(&request, &reqstat));
            }

            t_end = MPI_Wtime();
        }

        MPI_CHECK(MPI_Type_free(&type));

        latency = (t_end - t_start) * 1.0e6 / (2.0 * options.iterations);

        MPI_CHECK(MPI_Reduce(&latency, &total_lat, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD));

        avg_lat = total_lat/(double) (pairs * 2);

        if(0 == rank) {
            fprintf(stdout, "%-*d%-*d%-*d%*.*f\n",
                    10, size,
                    10, options.dt_block_size,
                    10, options.dt_stride_size,
                    FIELD_WIDTH, FLOAT_PRECISION, avg_lat);
            fflush(stdout);
        }
    }
}

/* vi: set sw=4 sts=4 tw=80: */
