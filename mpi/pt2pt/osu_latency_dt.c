#define BENCHMARK "OSU MPI%s Latency Test"
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

int
main (int argc, char *argv[])
{
    int myid, numprocs, i;
    int size;
    MPI_Request request;
    MPI_Status reqstat;
    char *s_buf, *r_buf;
    double t_start = 0.0, t_end = 0.0;
    int po_ret = 0;
    int rep_count;
    MPI_Datatype type;
    options.bench = PT2PT;
    options.subtype = LAT_DT;

    set_header(HEADER);
    set_benchmark_name("osu_latency_dt");

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
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

    if (0 == myid) {
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
                print_bad_usage_message(myid);
                break;
            case PO_HELP_MESSAGE:
                print_help_message(myid);
                break;
            case PO_VERSION_MESSAGE:
                print_version_message(myid);
                MPI_CHECK(MPI_Finalize());
                exit(EXIT_SUCCESS);
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

    if(numprocs != 2) {
        if(myid == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }

        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
        /* Error allocating memory */
        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    print_header(myid, LAT);

    /* Latency test */
    for(size = options.min_message_size; size <= options.max_message_size; size = (size ? size * 2 : 1)) {
        set_buffer_pt2pt(s_buf, myid, options.accel, 'a', size);
        set_buffer_pt2pt(r_buf, myid, options.accel, 'b', size);

        if(size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        /* Define DDT */
        rep_count = size / options.dt_block_size;
        MPI_CHECK(MPI_Type_vector(rep_count, options.dt_block_size, options.dt_stride_size, MPI_CHAR, &type));
        MPI_CHECK(MPI_Type_commit(&type));

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        if(myid == 0) {
            for(i = 0; i < options.iterations + options.skip; i++) {
                if(i == options.skip) {
                    t_start = MPI_Wtime();
                }

                MPI_CHECK(MPI_Isend(s_buf, 1, type, 1, 1, MPI_COMM_WORLD, &request));
                MPI_CHECK(MPI_Wait(&request, &reqstat));
                MPI_CHECK(MPI_Irecv(r_buf, 1, type, 1, 1, MPI_COMM_WORLD, &request));
                MPI_CHECK(MPI_Wait(&request, &reqstat));
            }

            t_end = MPI_Wtime();
        }

        else if(myid == 1) {
            for(i = 0; i < options.iterations + options.skip; i++) {
                MPI_CHECK(MPI_Irecv(r_buf, 1, type, 0, 1, MPI_COMM_WORLD, &request));
                MPI_CHECK(MPI_Wait(&request, &reqstat));
                MPI_CHECK(MPI_Isend(s_buf, 1, type, 0, 1, MPI_COMM_WORLD, &request));
                MPI_CHECK(MPI_Wait(&request, &reqstat));
            }
        }

        if(myid == 0) {
            double latency = (t_end - t_start) * 1e6 / (2.0 * options.iterations);

            fprintf(stdout, "%-*d%-*d%-*d%*.*f\n",
                    10, size, 
                    10, options.dt_block_size,
                    10, options.dt_stride_size,
                    FIELD_WIDTH, FLOAT_PRECISION, latency);
            fflush(stdout);
        }

        MPI_CHECK(MPI_Type_free(&type));
    }

    free_memory(s_buf, r_buf, myid);
    MPI_CHECK(MPI_Finalize());

    if (NONE != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}

