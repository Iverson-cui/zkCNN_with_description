//
// Created by 69029 on 3/16/2021.
//

#include "neuralNetwork.hpp"
#include "utils.hpp"
#include "global_var.hpp"
#include <polynomial.h>
#include <circuit.h>
#include <iostream>
#include <cmath>

using std::cerr;
using std::endl;
using std::ifstream;
using std::max;
using std::ofstream;

ifstream in;
ifstream conf;
ofstream out;

/**
 * This is neuralnetwork class constructor
 * In this constructor, only files are opened, no layers and weights are created
 * in->i_filename, conf->c_filename, out->o_filename
 * in, conf and out can be directly used and accessed in other member functions
 * weights loading is handed off to derived class constructors
 */
neuralNetwork::neuralNetwork(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, const string &i_filename,
                             const string &c_filename, const string &o_filename) : pic_size_x(psize_x), pic_size_y(psize_y), pic_channel(pchannel), pic_parallel(pparallel),
                                                                                   SIZE(0), NCONV_FAST_SIZE(1), NCONV_SIZE(2), FFT_SIZE(5),
                                                                                   AVE_POOL_SIZE(1), FC_SIZE(1), RELU_SIZE(1), act_ty(RELU_ACT)
{

    in.open(i_filename);
    if (!in.is_open())
        fprintf(stderr, "Can't find the input file!!!\n");
    conf.open(c_filename);
    if (!conf.is_open())
        fprintf(stderr, "Can't find the config file!!!\n");

    if (!o_filename.empty())
        out.open(o_filename);
}

/**
 * This is blank neural network constructor
 * This constructor doesn't open any files
 */
neuralNetwork::neuralNetwork(i64 psize, i64 pchannel, i64 pparallel, i64 kernel_size, i64 sec_size, i64 fc_size,
                             i64 start_channel, poolType pool_ty)
    : neuralNetwork(psize, psize, pchannel, pparallel, "", "", "")
{
    pool_bl = 2;
    pool_stride_bl = pool_bl >> 1;
    conv_section.resize(sec_size);

    convType conv_ty = kernel_size > 3 || pparallel > 1 ? FFT : NAIVE_FAST;
    i64 start = start_channel;
    for (i64 i = 0; i < sec_size; ++i)
    {
        conv_section[i].emplace_back(conv_ty, start << i, i ? (start << (i - 1)) : pic_channel, kernel_size);
        conv_section[i].emplace_back(conv_ty, start << i, start << i, kernel_size);
        pool.emplace_back(pool_ty, 2, 1);
    }

    i64 new_nx = (pic_size_x >> pool_stride_bl * conv_section.size());
    i64 new_ny = (pic_size_y >> pool_stride_bl * conv_section.size());
    for (i64 i = 0; i < fc_size; ++i)
        full_conn.emplace_back(i == fc_size - 1 ? 1000 : 4096, i ? 4096 : new_nx * new_ny * (start << (sec_size - 1)));
}

void neuralNetwork::create(prover &pr, bool only_compute)
{
    assert(pool.size() >= conv_section.size() - 1);

    // set memory layout for network
    initParam();
    // circuit initialization
    pr.C.init(Q_BIT_SIZE, SIZE);

    // val is vector<vector<F>>
    // first resize it based on number of layers
    pr.val.resize(SIZE);
    // turn these two vectors into iterators for easy access
    val = pr.val.begin();
    two_mul = pr.C.two_mul.begin();

    i64 layer_id = 0;
    // intialize input layer, create gates, calculate layer values
    inputLayer(pr.C.circuit[layer_id++]);

    // pic_size_x and y is the initial conv size
    new_nx_in = pic_size_x;
    new_ny_in = pic_size_y;
    // conv processing iteration
    for (i64 i = 0; i < conv_section.size(); ++i)
    {
        auto &sec = conv_section[i];
        for (i64 j = 0; j < sec.size(); ++j)
        {
            // conv is the atom conv kernel
            auto &conv = sec[j];
            // refresh conv parameters based on current conv kernel
            // because each time we are having different conv kernel
            refreshConvParam(new_nx_in, new_ny_in, conv);
            // pooling only appears after many conv layers
            pool_ty = i < pool.size() && j == sec.size() - 1 ? pool[i].ty : NONE;
            // quantization bit preservation
            x_bit = x_next_bit;
            // conv has different implementations
            // based on the conv type, different layers are created
            switch (conv.ty)
            {
            case FFT:
                paddingLayer(pr.C.circuit[layer_id], layer_id, conv.weight_start_id);
                fftLayer(pr.C.circuit[layer_id], layer_id);
                dotProdLayer(pr.C.circuit[layer_id], layer_id);
                ifftLayer(pr.C.circuit[layer_id], layer_id);
                addBiasLayer(pr.C.circuit[layer_id], layer_id, conv.bias_start_id);
                break;
            case NAIVE_FAST:
                naiveConvLayerFast(pr.C.circuit[layer_id], layer_id, conv.weight_start_id, conv.bias_start_id);
                break;
            // default: mult layer+add layer
            default:
                // initialize, create gates and calculate values for conv mult layer
                naiveConvLayerMul(pr.C.circuit[layer_id], layer_id, conv.weight_start_id);
                // initialize, create gates and calculate values for conv add layer
                naiveConvLayerAdd(pr.C.circuit[layer_id], layer_id, conv.bias_start_id);
            }

            // analyzes the value distribution from the previous layer to determine the optimal quantization bit-width for the current layer
            x_next_bit = getNextBit(layer_id - 1);
            // bit width adjustment between layers
            T = x_bit + w_bit - x_next_bit;
            Q_MAX = Q + T;
            // when max pooling is not used, relu is called
            if (pool_ty != MAX)
                reluActConvLayer(pr.C.circuit[layer_id], layer_id);
        }

        /**
         * There are two stages of conv.
         * For example, 20 conv operations can be divided into 4 stages, each stage contains 5 conv operations.
         * do the pooling after each 5 conv
         */
        if (i >= pool.size())
            continue;
        // calculate pooling parameters
        calcSizeAfterPool(pool[i]);
        // based on different pool type, do the pooling
        switch (pool[i].ty)
        {
        case AVG:
            avgPoolingLayer(pr.C.circuit[layer_id], layer_id);
            break;
        case MAX:
            maxPoolingLayer(pr.C, layer_id, pool[i].dcmp_start_id, pool[i].max_start_id,
                            pool[i].max_dcmp_start_id);
            break;
        }
    }

    pool_ty = NONE;
    for (int i = 0; i < full_conn.size(); ++i)
    {
        auto &fc = full_conn[i];
        refreshFCParam(fc);
        x_bit = x_next_bit;
        fullyConnLayer(pr.C.circuit[layer_id], layer_id, fc.weight_start_id, fc.bias_start_id);
        if (i == full_conn.size() - 1)
            break;

        // update the scale bit
        x_next_bit = getNextBit(layer_id - 1);
        T = x_bit + w_bit - x_next_bit;
        Q_MAX = Q + T;
        reluActFconLayer(pr.C.circuit[layer_id], layer_id);
    }

    assert(SIZE == layer_id);

    total_in_size += total_max_in_size + total_ave_in_size + total_relu_in_size;
    initLayer(pr.C.circuit[0], total_in_size, layerType::INPUT);
    assert(total_in_size == pr.val[0].size());

    printInfer(pr);
    //    printLayerValues(pr);

    if (only_compute)
        return;
    pr.C.initSubset();
    cerr << "finish creating circuit." << endl;
}

/**
 * initialized input layers
 * @param: circuit is the corresponding input layer
 */
void neuralNetwork::inputLayer(layer &circuit)
{
    // total_in_size is the input size, also the size of input layer
    initLayer(circuit, total_in_size, layerType::INPUT);

    // input layer only contains identity gates
    // create these gates
    for (i64 i = 0; i < total_in_size; ++i)
        // update uni_gates vector to contain these gates
        // (i,0,0,0) is passed to constructor of Unigate class
        // output pos=i, input pos=0, scaling factor=0, layer_id=0
        circuit.uni_gates.emplace_back(i, 0, 0, 0);

    // calculate output values of input layer
    calcInputLayer(circuit);
    printLayerInfo(circuit, 0);
}

void neuralNetwork::paddingLayer(layer &circuit, i64 &layer_id, i64 first_conv_id)
{
    i64 lenh = getFFTLen() >> 1;
    i64 size = lenh * channel_in * (pic_parallel + channel_out);
    initLayer(circuit, size, layerType::PADDING);
    circuit.fft_bit_length = getFFTBitLen();

    // data matrix
    i64 L = -padding;
    i64 Rx = nx_in + padding, Ry = ny_in + padding;

    for (i64 p = 0; p < pic_parallel; ++p)
        for (i64 ci = 0; ci < channel_in; ++ci)
            for (i64 x = L; x < Rx; ++x)
                for (i64 y = L; y < Ry; ++y)
                    if (check(x, y, nx_in, ny_in))
                    {
                        i64 g = cubIdx(p, ci, matIdx(Rx - x - 1, Ry - y - 1, ny_padded_in), channel_in, lenh);
                        i64 u = tesIdx(p, ci, x, y, channel_in, nx_in, ny_in);
                        circuit.uni_gates.emplace_back(g, u, layer_id - 1, 0);
                    }

    // kernel matrix
    i64 first = pic_parallel * channel_in * lenh;
    for (i64 co = 0; co < channel_out; ++co)
        for (i64 ci = 0; ci < channel_in; ++ci)
            for (i64 x = 0; x < nx_padded_in; ++x)
                for (i64 y = 0; y < ny_padded_in; ++y)
                    if (check(x, y, m, m))
                    {
                        i64 g = first + cubIdx(co, ci, matIdx(x, y, ny_padded_in), channel_in, lenh);
                        i64 u = first_conv_id + tesIdx(co, ci, x, y, channel_in, m, m);
                        circuit.uni_gates.emplace_back(g, u, 0, 0);
                    }

    readConvWeight(first_conv_id);
    calcNormalLayer(circuit, layer_id);
    printLayerInfo(circuit, layer_id++);
}

void neuralNetwork::fftLayer(layer &circuit, i64 &layer_id)
{
    i64 size = getFFTLen() * channel_in * (pic_parallel + channel_out);
    initLayer(circuit, size, layerType::FFT);
    circuit.fft_bit_length = getFFTBitLen();

    calcFFTLayer(circuit, layer_id);
    printLayerInfo(circuit, layer_id++);
}

void neuralNetwork::dotProdLayer(layer &circuit, i64 &layer_id)
{
    i64 len = getFFTLen();
    i64 size = len * channel_out * pic_parallel;
    initLayer(circuit, size, layerType::DOT_PROD);
    circuit.need_phase2 = true;
    circuit.fft_bit_length = getFFTBitLen();

    for (i64 p = 0; p < pic_parallel; ++p)
        for (i64 co = 0; co < channel_out; ++co)
            for (i64 ci = 0; ci < channel_in; ++ci)
            {
                i64 g = matIdx(p, co, channel_out);
                i64 u = matIdx(p, ci, channel_in);
                i64 v = matIdx(pic_parallel + co, ci, channel_in);
                circuit.bin_gates.emplace_back(g, u, v, 0, 1);
            }

    calcDotProdLayer(circuit, layer_id);
    printLayerInfo(circuit, layer_id++);
}

void neuralNetwork::ifftLayer(layer &circuit, i64 &layer_id)
{
    i64 len = getFFTLen(), lenh = len >> 1;
    i64 size = lenh * channel_out * pic_parallel;
    initLayer(circuit, size, layerType::IFFT);
    circuit.fft_bit_length = getFFTBitLen();
    F::inv(circuit.scale, F(1ULL << circuit.fft_bit_length));

    calcFFTLayer(circuit, layer_id);
    printLayerInfo(circuit, layer_id++);
}

void neuralNetwork::addBiasLayer(layer &circuit, i64 &layer_id, i64 first_bias_id)
{
    i64 len = getFFTLen();
    i64 size = nx_out * ny_out * channel_out * pic_parallel;
    initLayer(circuit, size, layerType::ADD_BIAS);

    i64 lenh = len >> 1;
    i64 L = -padding, Rx = nx_in + padding, Ry = ny_in + padding;
    for (i64 p = 0; p < pic_parallel; ++p)
        for (i64 co = 0; co < channel_out; ++co)
            for (i64 x = L; x + m <= Rx; x += (1 << log_stride))
                for (i64 y = L; y + m <= Ry; y += (1 << log_stride))
                {
                    i64 u = cubIdx(p, co, matIdx(Rx - x - 1, Ry - y - 1, ny_padded_in), channel_out, lenh);
                    i64 g = tesIdx(p, co, (x - L) >> log_stride, (y - L) >> log_stride, channel_out, nx_out, ny_out);
                    circuit.uni_gates.emplace_back(g, first_bias_id + co, 0, 0);
                    circuit.uni_gates.emplace_back(g, u, layer_id - 1, 0);
                }

    readBias(first_bias_id);
    calcNormalLayer(circuit, layer_id);
    printLayerInfo(circuit, layer_id++);
}

void neuralNetwork::naiveConvLayerFast(layer &circuit, i64 &layer_id, i64 first_conv_id, i64 first_bias_id)
{
    i64 size = nx_out * ny_out * channel_out * pic_parallel;
    initLayer(circuit, size, layerType::NCONV);
    circuit.need_phase2 = true;

    i64 L = -padding, Rx = nx_in + padding, Ry = ny_in + padding;
    i64 mat_in_size = nx_in * ny_in;
    i64 m_sqr = sqr(m);
    for (i64 p = 0; p < pic_parallel; ++p)
        for (i64 co = 0; co < channel_out; ++co)
            for (i64 ci = 0; ci < channel_in; ++ci)
                for (i64 x = L; x + m <= Rx; x += (1 << log_stride))
                    for (i64 y = L; y + m <= Ry; y += (1 << log_stride))
                    {
                        i64 g = tesIdx(p, co, ((x - L) >> log_stride), ((y - L) >> log_stride), channel_out, nx_out, ny_out);
                        if (ci == 0 && ~first_bias_id)
                            circuit.uni_gates.emplace_back(g, first_bias_id + co, 0, 0);
                        for (i64 tx = x; tx < x + m; ++tx)
                            for (i64 ty = y; ty < y + m; ++ty)
                                if (check(tx, ty, nx_in, ny_in))
                                {
                                    i64 u = tesIdx(p, ci, tx, ty, channel_in, nx_in, ny_in);
                                    i64 v = first_conv_id + tesIdx(co, ci, tx - x, ty - y, channel_in, m, m);
                                    circuit.bin_gates.emplace_back(g, u, v, 0, 2 * (u8)(layer_id > 1));
                                }
                    }

    readConvWeight(first_conv_id);
    if (~first_bias_id)
        readBias(first_bias_id);
    calcNormalLayer(circuit, layer_id);
    printLayerInfo(circuit, layer_id++);
}

/**
 * processes mult layer in naive conv layer
 * @param circuit: the layer to be processed
 * @param layer_id: the index of the layer in the whole circuit
 * create mult gates, read weights, do forward computation
 * forward results are stored in val[layer_id]
 */
void neuralNetwork::naiveConvLayerMul(layer &circuit, i64 &layer_id, i64 first_conv_id)
{
    // size of output matrix
    i64 mat_out_size = nx_out * ny_out;
    // size of input matrix
    i64 mat_in_size = nx_in * ny_in;
    // m=conv.size, kernel size
    i64 m_sqr = sqr(m);
    // L is left/top boundary with padding
    // Rx, Ry is right/bottom boundaries with padding
    i64 L = -padding, Rx = nx_in + padding, Ry = ny_in + padding;

    // g is the output size
    i64 g = 0;
    // this nested loop creates all multiplication gates, each for a weight and input multiplication
    // parallel processing units
    for (i64 p = 0; p < pic_parallel; ++p)
        // output channels
        for (i64 co = 0; co < channel_out; ++co)
            // input channels
            for (i64 ci = 0; ci < channel_in; ++ci)
                // x stride aware stepping
                for (i64 x = L; x + m <= Rx; x += (1 << log_stride))
                    // y stride aware stepping
                    for (i64 y = L; y + m <= Ry; y += (1 << log_stride))
                        // each conv window
                        for (i64 tx = x; tx < x + m; ++tx)
                            for (i64 ty = y; ty < y + m; ++ty)
                                // check if (tx,ty) is within input matrix boundary
                                if (check(tx, ty, nx_in, ny_in))
                                {
                                    // calculate linear index for input u
                                    i64 u = tesIdx(p, ci, tx, ty, channel_in, nx_in, ny_in);
                                    // calculate linear index for weight v
                                    i64 v = first_conv_id + tesIdx(co, ci, tx - x, ty - y, channel_in, m, m);
                                    // creates the multiplication gates with input u and v, output g, scaling 0
                                    // last argument is layer reference
                                    circuit.bin_gates.emplace_back(g++, u, v, 0, 2 * (u8)(layer_id > 1));
                                }

    // initialize this layer
    // for mult layer, this layer output size = number of multiplication gates
    initLayer(circuit, g, layerType::NCONV_MUL);
    circuit.need_phase2 = true;
    // read the conv weights from file and update val[0]
    readConvWeight(first_conv_id);
    // after loading the weights and creating all gates, we can do the forward computation
    calcNormalLayer(circuit, layer_id);
    // print layer info, currently does nothing
    printLayerInfo(circuit, layer_id++);
}

/**
 * processes add layer for naive conv layer
 */
void neuralNetwork::naiveConvLayerAdd(layer &circuit, i64 &layer_id, i64 first_bias_id)
{
    // size of output matrix
    i64 size = nx_out * ny_out * channel_out * pic_parallel;
    // initialize this layer
    // this layer output size is calculated based on output channels and dims etc
    initLayer(circuit, size, layerType::NCONV_ADD);
    i64 mat_in_size = nx_in * ny_in;
    i64 m_sqr = sqr(m);
    i64 L = -padding, Rx = nx_in + padding, Ry = ny_in + padding;

    i64 u = 0;
    // gate creation nested loop
    for (i64 p = 0; p < pic_parallel; ++p)
        for (i64 co = 0; co < channel_out; ++co)
            for (i64 ci = 0; ci < channel_in; ++ci)
                for (i64 x = L; x + m <= Rx; x += (1 << log_stride))
                    for (i64 y = L; y + m <= Ry; y += (1 << log_stride))
                    {
                        // each output position g, result is calculated by sum(input*weight)+bias
                        i64 g = tesIdx(p, co, ((x - L) >> log_stride), ((y - L) >> log_stride), channel_out, nx_out, ny_out);
                        i64 cnt = 0;
                        // bias addition, only once for each output position
                        if (ci == 0 && ~first_bias_id)
                        {
                            circuit.uni_gates.emplace_back(g, first_bias_id + co, 0, 0);
                            ++cnt;
                        }
                        // accumulate all input*weight products
                        // each iteration round of the following two loops produces one input*weight product
                        for (i64 tx = x; tx < x + m; ++tx)
                            for (i64 ty = y; ty < y + m; ++ty)
                                if (check(tx, ty, nx_in, ny_in))
                                {
                                    circuit.uni_gates.emplace_back(g, u++, layer_id - 1, 0);
                                    ++cnt;
                                }
                    }

    // load bias into val[0]
    if (~first_bias_id)
        readBias(first_bias_id);
    // calculate gate output values
    calcNormalLayer(circuit, layer_id);
    // print info
    printLayerInfo(circuit, layer_id++);
}

/**
 * Function for ReLU layers after convolutional layers
 * using bit decomposition to convert ReLU operation into a series of gates
 */
void neuralNetwork::reluActConvLayer(layer &circuit, i64 &layer_id)
{
    i64 mat_out_size = nx_out * ny_out;
    i64 size = 1L * mat_out_size * channel_out * (2 + Q_MAX) * pic_parallel;
    // number of output elements being processed by ReLU
    i64 block_len = mat_out_size * channel_out * pic_parallel;

    // size of all bit decomposition values
    i64 dcmp_cnt = block_len * Q_MAX;
    i64 first_dcmp_id = val[0].size();
    // extends val[0] to accommodate these new values
    val[0].resize(val[0].size() + dcmp_cnt);
    total_relu_in_size += dcmp_cnt;

    initLayer(circuit, size, layerType::RELU);
    circuit.need_phase2 = true;

    circuit.zero_start_id = block_len;

    /**
     * first gate creation loop
     * this loop takes input from layer 0
     * the prover provide auxiliary bit decomposition for all of the conv result elements
     * gates in this loop do the ReLU job, it outputs the ReLU(x) based on prover's aux input.
     * but whether aux input is correct is yet to be tested
     */
    for (i64 g = 0; g < block_len; ++g)
    {
        // bit representations of each element g are provided by prover as auxiliary inputs
        // each element's bit representation takes up Q_MAX positions, there are in total block_len*Q_MAX bits
        // sign_u is the starting position of the bit decomposition values for element g
        // think of first_dcmp_id as the starting position of all bit decomposition values, an offset
        i64 sign_u = first_dcmp_id + g * Q_MAX;
        // iterate through every bit position for this element
        for (i64 s = 1; s < Q; ++s)
        {
            // v is the corresponding position
            i64 v = sign_u + s;
            // uniGates with scaling Q-1-s and lu 0
            // assign different weights to different bit positions, notice that bit positions are at input layer, they are provided by the prover as inputs
            // for a given g, accumulate all v bits and assign sum to g
            // this is the magnitude version of original number
            circuit.uni_gates.emplace_back(g, v, 0, Q - 1 - s);

            /**
             * This binary gates has the following effects:
             * 1. If sign bit is 1, this number is negative, so output 0
             * 2. If sign bit is 0, this number is positive, so reconstructs positive magnitude for pos g
             */
            // scaling factor Q - s + Q_BIT_SIZE
            // lu 0
            circuit.bin_gates.emplace_back(g, sign_u, v, Q - s + Q_BIT_SIZE, 0);
        }
    }

    // calculating FFT parameters
    i64 len = getFFTLen();
    i64 lenh = len >> 1;
    // defining padding boundaries
    i64 L = -padding, Rx = nx_in + padding, Ry = ny_in + padding;

    /**
     * second gate creation loop
     * this loop iterates through every conv result in layer_id - 1
     * these gates are used to make sure the bit decomposition corresponds to the result numbers
     */

    for (i64 p = 0; p < pic_parallel; ++p)
        for (i64 co = 0; co < channel_out; ++co)
            for (i64 x = L; x + m <= Rx; x += (1 << log_stride))
                for (i64 y = L; y + m <= Ry; y += (1 << log_stride))
                {
                    // linear index of input
                    i64 u = tesIdx(p, co, (x - L) >> log_stride, (y - L) >> log_stride, channel_out, nx_out, ny_out);
                    // g is the gate output position in the circuit
                    // sign_v is the starting position of the bit decomposition values for element u
                    i64 g = block_len + u, sign_v = first_dcmp_id + u * Q_MAX;
                    circuit.uni_gates.emplace_back(g, u, layer_id - 1, Q_BIT_SIZE + 1);
                    circuit.bin_gates.emplace_back(g, u, sign_v, 1, 2 * (u8)(layer_id > 1));
                    prepareSignBit(layer_id - 1, u, sign_v);
                    for (i64 s = 1; s < Q_MAX; ++s)
                    {
                        i64 v = sign_v + s;
                        circuit.uni_gates.emplace_back(g, v, 0, Q_MAX - s - 1);
                        prepareDecmpBit(layer_id - 1, u, v, Q_MAX - s - 1);
                    }
                }

    /**
     * third gate creation loop
     * this loop are filled with squaring gates
     * gates in this loop is used to make sure all bit decomposition values are either 0 or 1
     */
    for (i64 g = block_len << 1; g < (block_len << 1) + block_len * Q_MAX; ++g)
    {
        i64 u = first_dcmp_id + g - (block_len << 1);
        circuit.bin_gates.emplace_back(g, u, u, 0, 0);
        circuit.uni_gates.emplace_back(g, u, 0, Q_BIT_SIZE + 1);
    }

    // calculate output values
    calcNormalLayer(circuit, layer_id);
    // print info
    printLayerInfo(circuit, layer_id++);
}

void neuralNetwork::reluActFconLayer(layer &circuit, i64 &layer_id)
{
    i64 block_len = channel_out * pic_parallel;
    i64 size = block_len * (2 + Q_MAX);
    initLayer(circuit, size, layerType::RELU);
    circuit.zero_start_id = block_len;
    circuit.need_phase2 = true;

    i64 dcmp_cnt = block_len * Q_MAX;
    i64 first_dcmp_id = val[0].size();
    val[0].resize(val[0].size() + dcmp_cnt);
    total_relu_in_size += dcmp_cnt;

    for (i64 g = 0; g < block_len; ++g)
    {
        i64 sign_u = first_dcmp_id + g * Q_MAX;
        for (i64 s = 1; s < Q; ++s)
        {
            i64 v = sign_u + s;
            circuit.uni_gates.emplace_back(g, v, 0, (Q - s - 1));
            circuit.bin_gates.emplace_back(g, sign_u, v, Q - s + Q_BIT_SIZE, 0);
        }
    }

    for (i64 u = 0; u < block_len; ++u)
    {
        i64 g = block_len + u, sign_v = first_dcmp_id + u * Q_MAX;
        circuit.uni_gates.emplace_back(g, u, layer_id - 1, Q_BIT_SIZE + 1);
        circuit.bin_gates.emplace_back(g, u, sign_v, 1, 2 * (u8)(layer_id > 1));
        prepareSignBit(layer_id - 1, u, sign_v);

        for (i64 s = 1; s < Q_MAX; ++s)
        {
            i64 v = sign_v + s;
            circuit.uni_gates.emplace_back(g, v, 0, Q_MAX - s - 1);
            prepareDecmpBit(layer_id - 1, u, v, Q_MAX - s - 1);
        }
    }

    for (i64 g = block_len << 1; g < (block_len << 1) + block_len * Q_MAX; ++g)
    {
        i64 u = first_dcmp_id + g - (block_len << 1);
        circuit.bin_gates.emplace_back(g, u, u, 0, 0);
        circuit.uni_gates.emplace_back(g, u, 0, Q_BIT_SIZE + 1);
    }

    calcNormalLayer(circuit, layer_id);
    printLayerInfo(circuit, layer_id++);
}

/**
 * This is the average pooling layer function
 */
void neuralNetwork::avgPoolingLayer(layer &circuit, i64 &layer_id)
{
    i64 mat_out_size = nx_out * ny_out;
    i64 zero_start_id = new_nx_in * new_ny_in * channel_out * pic_parallel;
    i64 size = zero_start_id + getPoolDecmpSize();
    u8 dpool_bl = pool_bl << 1;
    i64 pool_sz_sqr = sqr(pool_sz);
    initLayer(circuit, size, layerType::AVG_POOL);
    F::inv(circuit.scale, pool_sz_sqr);
    circuit.zero_start_id = zero_start_id;
    circuit.need_phase2 = true;

    i64 first_gate_id = val[0].size();
    val[0].resize(val[0].size() + zero_start_id * dpool_bl);
    total_ave_in_size += zero_start_id * dpool_bl;

    // [0 .. zero_start_id]
    // [zero_start_id .. zero_start_id + (g = 0..channel_out * mat_new_size) * dpool_bl + rm_i .. channel_out * mat_new_size * (1 + dpool_bl)]
    for (i64 p = 0; p < pic_parallel; ++p)
        for (i64 co = 0; co < channel_out; ++co)
            for (i64 x = 0; x + pool_sz <= nx_out; x += pool_stride)
                for (i64 y = 0; y + pool_sz <= ny_out; y += pool_stride)
                {
                    i64 g = tesIdx(p, co, (x >> pool_stride_bl), (y >> pool_stride_bl), channel_out, new_nx_in, new_ny_in);
                    F data = F_ZERO;
                    for (i64 tx = x; tx < x + pool_sz; ++tx)
                        for (i64 ty = y; ty < y + pool_sz; ++ty)
                        {
                            i64 u = tesIdx(p, co, tx, ty, channel_out, nx_out, ny_out);
                            circuit.uni_gates.emplace_back(g, u, layer_id - 1, 0);
                            data = data + val[layer_id - 1][u];
                        }

                    for (i64 rm_i = 0; rm_i < dpool_bl; ++rm_i)
                    {
                        i64 idx = matIdx(g, rm_i, dpool_bl), u = first_gate_id + idx, g_bit = zero_start_id + idx;
                        circuit.uni_gates.emplace_back(g, u, 0, dpool_bl - rm_i + Q_BIT_SIZE);
                        prepareFieldBit(F(data), u, dpool_bl - rm_i - 1);

                        // check bit
                        circuit.bin_gates.emplace_back(g_bit, u, u, 0, 0);
                        circuit.uni_gates.emplace_back(g_bit, u, 0, Q_BIT_SIZE + 1);
                    }
                }

    calcNormalLayer(circuit, layer_id);
    printLayerInfo(circuit, layer_id++);
}

void neuralNetwork::maxPoolingLayer(layeredCircuit &C, i64 &layer_id, i64 first_dcmp_id, i64 first_max_id,
                                    i64 first_max_dcmp_id)
{
    i64 mat_out_size = nx_out * ny_out;
    i64 tot_out_size = mat_out_size * channel_out * pic_parallel;
    i64 mat_new_size = new_nx_in * new_ny_in;
    i64 tot_new_size = mat_new_size * channel_out * pic_parallel;
    i64 pool_sz_sqr = sqr(pool_sz);

    i64 dcmp_cnt = getPoolDecmpSize();
    first_dcmp_id = val[0].size();
    val[0].resize(val[0].size() + dcmp_cnt);
    total_max_in_size += dcmp_cnt;

    i64 max_cnt = tot_new_size;
    first_max_id = val[0].size();
    val[0].resize(val[0].size() + max_cnt);
    total_max_in_size += max_cnt;

    i64 max_dcmp_cnt = tot_new_size * (Q_MAX - 1);
    first_max_dcmp_id = val[0].size();
    val[0].resize(val[0].size() + max_dcmp_cnt);
    total_max_in_size += max_dcmp_cnt;

    // 0:   max - everyone & max - (max bits) == 0
    //      [0..tot_new_size * sqr(pool_sz)][tot_new_size * sqr(pool_sz)..tot_new_size * sqr(pool_sz) + tot_new_size]
    i64 size_0 = tot_new_size * pool_sz_sqr + tot_new_size;
    layer &circuit = C.circuit[layer_id];
    initLayer(circuit, size_0, layerType::MAX_POOL);
    circuit.zero_start_id = tot_new_size * pool_sz_sqr;
    i64 fft_len = getFFTLen(), fft_lenh = fft_len >> 1;
    for (i64 p = 0; p < pic_parallel; ++p)
        for (i64 co = 0; co < channel_out; ++co)
        {
            for (i64 x = 0; x + pool_sz <= nx_out; x += pool_stride)
                for (i64 y = 0; y + pool_sz <= ny_out; y += pool_stride)
                {
                    i64 i_max = tesIdx(p, co, x >> pool_stride_bl, y >> pool_stride_bl, channel_out, new_nx_in, new_ny_in);
                    i64 u_max = first_max_id + i_max;
                    for (i64 tx = x; tx < x + pool_sz; ++tx)
                        for (i64 ty = y; ty < y + pool_sz; ++ty)
                        {
                            i64 g = cubIdx(tesIdx(p, co, x >> pool_stride_bl, y >> pool_stride_bl, channel_out, new_nx_in, new_ny_in), tx - x, ty - y, pool_sz, pool_sz);
                            i64 u_g = tesIdx(p, co, tx, ty, channel_out, nx_out, ny_out);
                            circuit.uni_gates.emplace_back(g, u_max, 0, 0);
                            circuit.uni_gates.emplace_back(g, u_g, layer_id - 1, Q_BIT_SIZE + 1);
                            prepareMax(layer_id - 1, u_g, u_max);
                        }
                }
        }

    for (i64 i_new = 0; i_new < tot_new_size; ++i_new)
    {
        i64 g_new = circuit.zero_start_id + i_new;
        i64 u_new = first_max_id + i_new;
        circuit.uni_gates.emplace_back(g_new, u_new, 0, Q_BIT_SIZE + 1);
        for (i64 i_new_bit = 0; i_new_bit < Q_MAX - 1; ++i_new_bit)
        {
            i64 u_new_bit = first_max_dcmp_id + matIdx(i_new, i_new_bit, Q_MAX - 1);
            circuit.uni_gates.emplace_back(g_new, u_new_bit, 0, Q_MAX - 2 - i_new_bit);
            prepareDecmpBit(0, u_new, u_new_bit, Q_MAX - 2 - i_new_bit);
        }
    }
    calcNormalLayer(circuit, layer_id);
    printLayerInfo(circuit, layer_id++);

    // 1:   (max - someone)^2 & max - everyone - ((max - everyone) bits) == 0
    //      [0..tot_new_size * (sqr(pool_sz) + 1 >> 1)][tot_new_size * (sqr(pool_sz) + 1 >> 1)..tot_new_size * (sqr(pool_sz) + 1 >> 1) + tot_new_size * sqr(pool_sz)]
    // 2:   (max - someone)^4
    // ?:   (max - someone)^(2^?)
    //      [0..(((tot_out_size + 1) / 2 + 1) / 2...+ 1) / 2]
    // f:   new tensor & (max - someone)^(pool_sz^2 + 1) & all (include minus and max) bits check
    //      [0..tot_new_size]
    //      [tot_new_size..tot_new_size * 2]
    //      [tot_new_size * 2..tot_new_size * (Q + 1)]
    //      [tot_new_size * (Q + 1)
    //          ..tot_new_size * (Q + 1) + (g = 0..tot_out_size) * (Q - 1) + bit_i
    //          ..tot_new_size * (Q + 1) + tot_new_size * (pool_sz^2) * (Q - 1)]
    i64 contain_max_ly = 1, ksize = pool_sz_sqr;
    while (!(ksize & 1))
    {
        ksize >>= 1;
        ++contain_max_ly;
    }
    ksize = pool_sz_sqr;

    for (int i = 1; i < pool_layer_cnt; ++i)
    {
        layer &circuit = C.circuit[layer_id];
        i64 size = tot_new_size * (((ksize + 1) >> 1) + (i64)(i == 1) * ksize) +
                   (i64)(i == pool_layer_cnt - 1) * tot_new_size * Q_MAX +
                   (i64)(i == pool_layer_cnt - 1) * tot_new_size * pool_sz_sqr * (Q_MAX - 1);
        initLayer(circuit, size, layerType::MAX_POOL);
        circuit.need_phase2 = true;

        // new tensor
        i64 before_mul = 0;
        if (i == pool_layer_cnt - 1)
        {
            before_mul = tot_new_size;
            for (i64 g = 0; g < tot_new_size; ++g)
                for (i64 j = 0; j < Q - 1; ++j)
                {
                    i64 u = first_max_dcmp_id + matIdx(g, j, Q_MAX - 1);
                    circuit.uni_gates.emplace_back(g, u, 0, Q - 2 - j);
                }
        }

        // multiplications of subtraction
        for (i64 cnt = 0; cnt < tot_new_size; ++cnt)
        {
            i64 v_max = first_max_id + cnt;
            for (i64 j = 0; (j << 1) < ksize; ++j)
            {
                i64 idx = matIdx(cnt, j, (ksize + 1) >> 1);
                i64 g = before_mul + idx;
                i64 u = matIdx(cnt, (j << 1), ksize);
                if ((j << 1 | 1) < ksize)
                {
                    i64 v = matIdx(cnt, (j << 1 | 1), ksize);
                    circuit.bin_gates.emplace_back(g, u, v, 0, layer_id > 1);
                }
                else if (i == contain_max_ly)
                    circuit.bin_gates.emplace_back(g, u, v_max, 0, 2 * (u8)(layer_id > 1));
                else
                    circuit.uni_gates.emplace_back(g, u, layer_id - 1, 0);
            }
        }

        if (i == 1)
        {
            i64 minus_cnt = tot_new_size * ksize;
            i64 minus_new_cnt = tot_new_size * ((ksize + 1) >> 1);
            circuit.zero_start_id = minus_new_cnt;
            for (i64 v = 0; v < minus_cnt; ++v)
            {
                i64 g = minus_new_cnt + v;
                circuit.uni_gates.emplace_back(g, v, layer_id - 1, Q_BIT_SIZE + 1);
                for (i64 bit_j = 0; bit_j < Q_MAX - 1; ++bit_j)
                {
                    i64 u = first_dcmp_id + matIdx(v, bit_j, Q_MAX - 1);
                    circuit.uni_gates.emplace_back(g, u, 0, Q_MAX - 2 - bit_j);
                    prepareDecmpBit(layer_id - 1, v, u, Q_MAX - 2 - bit_j);
                }
            }
        }
        else if (i == pool_layer_cnt - 1)
        {
            i64 minus_cnt = tot_new_size * pool_sz_sqr;
            circuit.zero_start_id = before_mul;
            for (i64 j = 0; j < minus_cnt; ++j)
            {
                i64 g = before_mul + tot_new_size + j;
                i64 u = first_dcmp_id + j;
                circuit.bin_gates.emplace_back(g, u, u, 0, 0);
                circuit.uni_gates.emplace_back(g, u, 0, Q_BIT_SIZE + 1);
            }
        }
        ksize = (ksize + 1) >> 1;
        calcNormalLayer(circuit, layer_id);
        printLayerInfo(circuit, layer_id++);
    }
}

void neuralNetwork::fullyConnLayer(layer &circuit, i64 &layer_id, i64 first_fc_id, i64 first_bias_id)
{
    i64 size = channel_out * pic_parallel;
    initLayer(circuit, size, layerType::FCONN);
    circuit.need_phase2 = true;

    for (i64 p = 0; p < pic_parallel; ++p)
        for (i64 co = 0; co < channel_out; ++co)
        {
            i64 g = matIdx(p, co, channel_out);
            circuit.uni_gates.emplace_back(g, first_bias_id + co, 0, 0);
            for (i64 ci = 0; ci < channel_in; ++ci)
            {
                i64 u = matIdx(p, ci, channel_in);
                i64 v = first_fc_id + matIdx(co, ci, channel_in);
                circuit.bin_gates.emplace_back(g, u, v, 0, 2 * (u8)(layer_id > 1));
            }
        }

    readFconWeight(first_fc_id);
    readBias(first_bias_id);
    calcNormalLayer(circuit, layer_id);
    printLayerInfo(circuit, layer_id++);
}

void neuralNetwork::refreshConvParam(i64 new_nx, i64 new_ny, const convKernel &conv)
{
    // original input dimensions
    nx_in = new_nx;
    ny_in = new_ny;
    // padding size
    padding = conv.padding;
    // calculate padded input dimensions
    nx_padded_in = nx_in + (conv.padding * 2);
    ny_padded_in = ny_in + (conv.padding * 2);

    // extract information from convKernel
    m = conv.size;
    channel_in = conv.channel_in;
    channel_out = conv.channel_out;
    log_stride = conv.stride_bl;

    // output dimension calculation
    nx_out = ((nx_padded_in - m) >> log_stride) + 1;
    ny_out = ((ny_padded_in - m) >> log_stride) + 1;

    // output dimensions become input dimensions for next layer
    new_nx_in = nx_out;
    new_ny_in = ny_out;
    conv_layer_cnt = conv.ty == FFT ? FFT_SIZE : conv.ty == NAIVE ? NCONV_SIZE
                                                                  : NCONV_FAST_SIZE;
}

void neuralNetwork::refreshFCParam(const fconKernel &fc)
{
    nx_in = nx_out = m = 1;
    ny_in = ny_out = 1;
    channel_in = fc.channel_in;
    channel_out = fc.channel_out;
}

i64 neuralNetwork::getFFTLen() const
{
    return 1L << getFFTBitLen();
}

i8 neuralNetwork::getFFTBitLen() const
{
    return ceilPow2BitLength((u32)nx_padded_in * ny_padded_in) + 1;
}

// input:   [data]
//          [[conv_kernel || relu_conv_bit_decmp]{sec.size()}[max_pool]{if maxPool}[pool_bit_decmp]]{conv_section.size()}
//          [fc_kernel || relu_fc_bit_decmp]
/**
 * Sets the memory layout for the data and weights
 * point out where does data and weights begin, but don't load for now
 */
void neuralNetwork::initParam()
{
    act_layer_cnt = RELU_SIZE;
    i64 total_conv_layer_cnt = 0, total_pool_layer_cnt = 0;
    total_in_size = 0;
    total_para_size = 0;
    total_relu_in_size = 0;
    total_ave_in_size = 0;
    total_max_in_size = 0;

    // data
    // everything before pos is input data
    i64 pos = pic_size_x * pic_size_y * pic_channel * pic_parallel;

    new_nx_in = pic_size_x;
    new_ny_in = pic_size_y;
    // conv layers memory layout
    for (i64 i = 0; i < conv_section.size(); ++i)
    {
        auto &sec = conv_section[i];
        for (i64 j = 0; j < sec.size(); ++j)
        {
            refreshConvParam(new_nx_in, new_ny_in, sec[j]);
            // conv_kernel
            // everything after pos is network parameters
            // weights
            sec[j].weight_start_id = pos;
            u32 para_size = sqr(m) * channel_in * channel_out;
            pos += para_size;
            total_para_size += para_size;
            fprintf(stderr, "kernel weight: %11d%11lld\n", para_size, total_para_size);

            // bias
            sec[j].bias_start_id = pos;
            pos += channel_out;
            total_para_size += channel_out;
            fprintf(stderr, "bias   weight: %11lld%11lld\n", channel_out, total_para_size);
        }

        total_conv_layer_cnt += sec.size() * (conv_layer_cnt + act_layer_cnt);

        if (i >= pool.size())
            continue;
        calcSizeAfterPool(pool[i]);
        total_pool_layer_cnt += pool_layer_cnt;
        if (pool[i].ty == MAX)
            if (act_ty == RELU_ACT)
                total_conv_layer_cnt -= act_layer_cnt;
    }

    // fully connected layer memory layout
    for (int i = 0; i < full_conn.size(); ++i)
    {
        auto &fc = full_conn[i];
        refreshFCParam(fc);
        // fc_kernel
        fc.weight_start_id = pos;
        u32 para_size = channel_out * channel_in;
        pos += para_size;
        total_para_size += para_size;
        fprintf(stderr, "kernel weight: %11d%11lld\n", para_size, total_para_size);
        fc.bias_start_id = pos;
        pos += channel_out;
        total_para_size += channel_out;
        fprintf(stderr, "bias   weight: %11lld%11lld\n", channel_out, total_para_size);
        if (i == full_conn.size() - 1)
            break;
    }
    total_in_size = pos;

    SIZE = 1 + total_conv_layer_cnt + total_pool_layer_cnt + (FC_SIZE + RELU_SIZE) * full_conn.size();
    if (!full_conn.empty())
        SIZE -= RELU_SIZE;
    cerr << "SIZE: " << SIZE << endl;
}

/**
 * This function is for printing layer information after forwarding
 * but now it does nothing, all commented out
 */
void neuralNetwork::printLayerInfo(const layer &circuit, i64 layer_id)
{
    //    fprintf(stderr, "+ %2lld " , layer_id);
    //    switch (circuit.ty) {
    //        case layerType::INPUT:         fprintf(stderr, "inputLayer         ");  break;
    //        case layerType::PADDING:       fprintf(stderr, "paddingLayer       ");  break;
    //        case layerType::FFT:           fprintf(stderr, "fftLayer           ");  break;
    //        case layerType::DOT_PROD:      fprintf(stderr, "dotProdLayer       ");  break;
    //        case layerType::IFFT:          fprintf(stderr, "ifftLayer          ");  break;
    //        case layerType::ADD_BIAS:      fprintf(stderr, "addBiasLayer       ");  break;
    //        case layerType::RELU:          fprintf(stderr, "reluActLayer       ");  break;
    //        case layerType::Sqr:           fprintf(stderr, "squareActLayer     ");  break;
    //        case layerType::OPT_AVG_POOL:  fprintf(stderr, "avgOptPoolingLayer ");  break;
    //        case layerType::AVG_POOL:      fprintf(stderr, "avgPoolingLayer    ");  break;
    //        case layerType::MAX_POOL:      fprintf(stderr, "maxPoolingLayer    ");  break;
    //        case layerType::FCONN:         fprintf(stderr, "fullyConnLayer     ");  break;
    //        case layerType::NCONV:         fprintf(stderr, "naiveConvFast      ");  break;
    //        case layerType::NCONV_MUL:     fprintf(stderr, "naiveConvMul       ");  break;
    //        case layerType::NCONV_ADD:     fprintf(stderr, "naiveConvAdd       ");  break;
    // m
    //    }
    //    fprintf(stderr, "%11u (2^%2d)\n", circuit.size, (int) circuit.bit_length);
}

void neuralNetwork::printWitnessInfo(const layer &circuit) const
{
    assert(circuit.size == total_in_size);
    u32 total_data_in_size = total_in_size - total_relu_in_size - total_ave_in_size - total_max_in_size;
    fprintf(stderr, "%u (2^%2d) = %u (%.2f%% data) + %lld (%.2f%% relu) + %lld (%.2f%% ave) + %lld (%.2f%% max), ",
            circuit.size, circuit.bit_length, total_data_in_size, 100.0 * total_data_in_size / (double)total_in_size,
            total_relu_in_size, 100.0 * total_relu_in_size / (double)total_in_size,
            total_ave_in_size, 100.0 * total_ave_in_size / (double)total_in_size,
            total_max_in_size, 100.0 * total_max_in_size / (double)total_in_size);
    output_tb[WS_OUT_ID] = std::to_string(circuit.size) + "(2^" + std::to_string(ceilPow2BitLength(circuit.size)) + ")";
}

i64 neuralNetwork::getPoolDecmpSize() const
{
    switch (pool_ty)
    {
    case AVG:
        return new_nx_in * new_ny_in * (pool_bl << 1) * channel_out * pic_parallel;
    case MAX:
        return new_nx_in * new_ny_in * sqr(pool_sz) * channel_out * pic_parallel * (Q_MAX - 1);
    default:
        assert(false);
    }
}

/**
 * This function calculates the parameters and output dimensions for pooling layers in a neural network
 * @param: p is the poolKernel object containing pooling parameters
 */
void neuralNetwork::calcSizeAfterPool(const poolKernel &p)
{
    // extract information from poolKernel
    pool_sz = p.size;
    pool_bl = ceilPow2BitLength(pool_sz);
    pool_stride_bl = p.stride_bl;
    pool_stride = 1 << p.stride_bl;
    // based on different pool type, number of pool layers is different
    pool_layer_cnt = p.ty == MAX ? 1 + ceilPow2BitLength(sqr(p.size) + 1) : AVE_POOL_SIZE;
    // calculate output dimensions, i.e. dims after pooling
    new_nx_in = ((nx_out - pool_sz) >> pool_stride_bl) + 1;
    new_ny_in = ((ny_out - pool_sz) >> pool_stride_bl) + 1;
}

/**
 * read input layer data, quantize them and store in val[0]
 * @param: circuit is the input layer
 */
void neuralNetwork::calcInputLayer(layer &circuit)
{
    // resizing val[0], input layer gates output, to fit input data
    // but now val[0] is still all 0
    val[0].resize(circuit.size);

    assert(val[0].size() == total_in_size);
    // sets up an iterator
    auto val_0 = val[0].begin();

    // preset the mx and mn values
    // num is the temp value of numbers being read now
    double num, mx = -10000, mn = 10000;
    // input_dat stores all the input data
    vector<double> input_dat;
    // update input_dat in iterations
    for (i64 ci = 0; ci < pic_channel; ++ci)
        for (i64 x = 0; x < pic_size_x; ++x)
            for (i64 y = 0; y < pic_size_y; ++y)
            {
                // in has already been set to input file in constructor
                // assign the now value of in to num
                in >> num;
                // update input_dat
                input_dat.push_back(num);
                // based on num, update mx and mn
                mx = max(mx, num);
                mn = min(mn, num);
            }

    // (mx - mn) * 2^i <= 2^Q - 1
    // quant_shr = i
    // x_next_bit here is the i in the formula above
    x_next_bit = (int)(log(((1 << (Q - 1)) - 1) / (mx - mn)) / log(2));
    // make sure the formula is satisfied
    if ((int)((mx - mn) * exp2(x_next_bit)) > (1 << (Q - 1)) - 1)
        --x_next_bit;

    // quantization process
    for (i64 p = 0; p < pic_parallel; ++p)
    {
        i64 i = 0;
        for (i64 ci = 0; ci < pic_channel; ++ci)
            for (i64 x = 0; x < pic_size_x; ++x)
                for (i64 y = 0; y < pic_size_y; ++y)
                    // quantize each value by multiplying 2^x_next_bit and rounding
                    // write quantized data to val_0
                    *val_0++ = F((i64)(input_dat[i++] * exp2(x_next_bit)));
    }
    // clean up operation to avoid garbage values
    for (; val_0 < val[0].begin() + circuit.size; ++val_0)
        val_0->clear();
}

/**
 * reads convolutional layer weights, store them in val[0] and quantize them
 */
void neuralNetwork::readConvWeight(i64 first_conv_id)
{
    // val_0 is an iterator pointing to val[0] container
    // modifying val_0 means modifying val[0]
    // later val[0] will be used
    auto val_0 = val[0].begin() + first_conv_id;

    // default max and min values
    double num, mx = -10000, mn = 10000;
    vector<double> input_dat;
    for (i64 co = 0; co < channel_out; ++co)
        for (i64 ci = 0; ci < channel_in; ++ci)
            for (i64 x = 0; x < m; ++x)
                for (i64 y = 0; y < m; ++y)
                {
                    in >> num;
                    input_dat.push_back(num);
                    mx = max(mx, num);
                    mn = min(mn, num);
                }

    // (mx - mn) * 2^i <= 2^Q - 1
    // quant_shr = i
    // w_bit is the quantization bit for weights
    w_bit = (int)(log(((1 << (Q - 1)) - 1) / (mx - mn)) / log(2));
    if ((int)((mx - mn) * exp2(w_bit)) > (1 << (Q - 1)) - 1)
        --w_bit;

    // val_0 contains values after quantization
    for (double i : input_dat)
        *val_0++ = F((i64)(i * exp2(w_bit)));
}

// read bias parameters and load into val[0]
void neuralNetwork::readBias(i64 first_bias_id)
{
    auto val_0 = val[0].begin() + first_bias_id;

    double num, mx = -10000, mn = 10000;
    vector<double> input_dat;
    for (i64 co = 0; co < channel_out; ++co)
    {
        in >> num;
        input_dat.push_back(num);
        mx = max(mx, num);
        mn = min(mn, num);
    }

    for (double i : input_dat)
        *val_0++ = F((i64)(i * exp2(w_bit + x_bit)));
}

void neuralNetwork::readFconWeight(i64 first_fc_id)
{
    double num, mx = -10000, mn = 10000;
    auto val_0 = val[0].begin() + first_fc_id;

    vector<double> input_dat;
    for (i64 co = 0; co < channel_out; ++co)
        for (i64 ci = 0; ci < channel_in; ++ci)
        {
            in >> num;
            input_dat.push_back(num);
            mx = max(mx, num);
            mn = min(mn, num);
        }

    // (mx - mn) * 2^i <= 2^Q - 1
    // quant_shr = i
    w_bit = (int)(log(((1 << (Q - 1)) - 1) / (mx - mn)) / log(2));
    if ((int)((mx - mn) * exp2(w_bit)) > (1 << (Q - 1)) - 1)
        --w_bit;

    for (double i : input_dat)
        *val_0++ = F((i64)(i * exp2(w_bit)));
}

void neuralNetwork::prepareDecmpBit(i64 layer_id, i64 idx, i64 dcmp_id, i64 bit_shift)
{
    auto data = abs(val[layer_id].at(idx).getInt64());
    val[0].at(dcmp_id) = (data >> bit_shift) & 1;
}

void neuralNetwork::prepareFieldBit(const F &data, i64 dcmp_id, i64 bit_shift)
{
    auto tmp = abs(data.getInt64());
    val[0].at(dcmp_id) = (tmp >> bit_shift) & 1;
}

void neuralNetwork::prepareSignBit(i64 layer_id, i64 idx, i64 dcmp_id)
{
    val[0].at(dcmp_id) = val[layer_id].at(idx).isNegative() ? F_ONE : F_ZERO;
}

void neuralNetwork::prepareMax(i64 layer_id, i64 idx, i64 max_id)
{
    auto data = val[layer_id].at(idx).isNegative() ? F_ZERO : val[layer_id].at(idx);
    if (data > val[0].at(max_id))
        val[0].at(max_id) = data;
}

/**
 * forward computation for a NN layer
 * @param circuit: the layer to be computed
 * @param layer_id: the index of the layer in the whole circuit
 * the output of the layer will be stored in val[layer_id]
 * this function only supports eq gates and mult gates
 */
void neuralNetwork::calcNormalLayer(const layer &circuit, i64 layer_id)
{
    // allocate memory for output values
    val[layer_id].resize(circuit.size);
    // initialize all values to 0
    for (auto &x : val[layer_id])
        x.clear();

    // iterate through all unary gates
    for (auto &gate : circuit.uni_gates)
    {
        // input is val[gate.lu].at(gate.u), meaning gate.u index of gate.lu layer
        // scaling factor is two_mul[gate.sc]
        val[layer_id].at(gate.g) = val[layer_id].at(gate.g) + val[gate.lu].at(gate.u) * two_mul[gate.sc];
    }

    // iterate through all binary gates
    for (auto &gate : circuit.bin_gates)
    {
        // get u,v layer reference
        u8 bin_lu = gate.getLayerIdU(layer_id), bin_lv = gate.getLayerIdV(layer_id);
        // update output value by multiplying two input values and gate scaling factor
        val[layer_id].at(gate.g) = val[layer_id].at(gate.g) + val[bin_lu].at(gate.u) * val[bin_lv][gate.v] * two_mul[gate.sc];
    }

    F mx_val = F_ZERO, mn_val = F_ZERO;
    // final layer scaling for all output values based on circuit.scale
    for (i64 g = 0; g < circuit.size; ++g)
        val[layer_id].at(g) = val[layer_id].at(g) * circuit.scale;
}

void neuralNetwork::calcDotProdLayer(const layer &circuit, i64 layer_id)
{
    val[layer_id].resize(circuit.size);
    for (int i = 0; i < circuit.size; ++i)
        val[layer_id][i].clear();

    char fft_bit = circuit.fft_bit_length;
    u32 fft_len = 1 << fft_bit;
    u8 l = layer_id - 1;
    for (auto &gate : circuit.bin_gates)
        for (int s = 0; s < fft_len; ++s)
            val[layer_id][gate.g << fft_bit | s] = val[layer_id][gate.g << fft_bit | s] +
                                                   val[l][gate.u << fft_bit | s] * val[l][gate.v << fft_bit | s];
}

void neuralNetwork::calcFFTLayer(const layer &circuit, i64 layer_id)
{
    i64 fft_len = 1ULL << circuit.fft_bit_length;
    i64 fft_lenh = fft_len >> 1;
    val[layer_id].resize(circuit.size);
    std::vector<F> arr(fft_len, F_ZERO);
    if (circuit.ty == layerType::FFT)
        for (i64 c = 0, d = 0; d < circuit.size; c += fft_lenh, d += fft_len)
        {
            for (i64 j = c; j < c + fft_lenh; ++j)
                arr[j - c] = val[layer_id - 1].at(j);
            for (i64 j = fft_lenh; j < fft_len; ++j)
                arr[j].clear();
            fft(arr, circuit.fft_bit_length, circuit.ty == layerType::IFFT);
            for (i64 j = d; j < d + fft_len; ++j)
                val[layer_id].at(j) = arr[j - d];
        }
    else
        for (u32 c = 0, d = 0; c < circuit.size; c += fft_lenh, d += fft_len)
        {
            for (i64 j = d; j < d + fft_len; ++j)
                arr[j - d] = val[layer_id - 1].at(j);
            fft(arr, circuit.fft_bit_length, circuit.ty == layerType::IFFT);
            for (i64 j = c; j < c + fft_lenh; ++j)
                val[layer_id].at(j) = arr[j - c];
        }
}

int neuralNetwork::getNextBit(int layer_id)
{
    F mx = F_ZERO, mn = F_ZERO;
    for (const auto &x : val[layer_id])
    {
        if (!x.isNegative())
            mx = max(mx, x);
        else
            mn = max(mn, -x);
    }
    i64 x = (mx + mn).getInt64();
    double real_scale = x / exp2(x_bit + w_bit);
    int res = (int)log2(((1 << (Q - 1)) - 1) / real_scale);
    return res;
}

void neuralNetwork::printLayerValues(prover &pr)
{
    for (i64 i = 0; i < SIZE; ++i)
    {
        //        if (pr.C.circuit[i].ty == layerType::FCONN || pr.C.circuit[i].ty == layerType::ADD_BIAS || i && i < SIZE - 1 && pr.C.circuit[i + 1].ty == layerType::PADDING) {
        cerr << i << "(" << pr.C.circuit[i].zero_start_id << ", " << pr.C.circuit[i].size << "):\t";
        for (i64 j = 0; j < std::min(200u, pr.C.circuit[i].size); ++j)
            if (!pr.val[i][j].isZero())
                cerr << pr.val[i][j] << ' ';
        cerr << endl;
        for (i64 j = pr.C.circuit[i].zero_start_id; j < pr.C.circuit[i].size; ++j)
            if (pr.val[i].at(j) != F_ZERO)
            {
                cerr << "WRONG! " << i << ' ' << j << ' ' << (-pr.val[i][j] * F_ONE) << endl;
                exit(EXIT_FAILURE);
            }
    }
}

void neuralNetwork::printInfer(prover &pr)
{
    // output the inference result with the size of (pic_parallel x n_class)
    if (out.is_open())
    {
        int n_class = full_conn.back().channel_out;
        for (int p = 0; p < pic_parallel; ++p)
        {
            int k = -1;
            F v;
            for (int c = 0; c < n_class; ++c)
            {
                auto tmp = val[SIZE - 1].at(matIdx(p, c, n_class));
                if (!tmp.isNegative() && (k == -1 || v < tmp))
                {
                    k = c;
                    v = tmp;
                }
            }
            out << k << endl;

            // output one-hot
            //            for (int c = 0; c < n_class; ++c) out << (k == c) << ' ';
            //            out << endl;
        }
    }
    out.close();
    printWitnessInfo(pr.C.circuit[0]);
}