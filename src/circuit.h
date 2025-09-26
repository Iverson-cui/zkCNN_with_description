#pragma once

#include <vector>
#include <unordered_map>
#include <utility>
#include <hyrax-bls12-381/polyCommit.hpp>
#include <unordered_set>
#include <iostream>
#include "global_var.hpp"

using std::cerr;
using std::endl;
using std::vector;

struct uniGate
{
    // g means output position, u means input position
    u32 g, u;
    // sc means scaling factor used to look up in the two_mul array
    // lu means layer id of input u
    u8 lu, sc;
    // struct constructor
    // this constructor just accept 4 params and assign them to the member variables
    uniGate(u32 _g, u32 _u, u8 _lu, u8 _sc) : g(_g), u(_u), lu(_lu), sc(_sc)
    {
        //        cerr << "uni: " << g << ' ' << u << ' ' << lu <<' ' << sc.real << endl;
    }
};

struct binGate
{
    // binary gates have two inputs u and v
    u32 g, u, v;
    // scaling factor and layer reference
    u8 sc, l;
    binGate(u32 _g, u32 _u, u32 _v, u8 _sc, u8 _l) : g(_g), u(_u), v(_v), sc(_sc), l(_l)
    {
        //        cerr << "bin: " << g << ' ' << u << ' ' << lu << ' ' << v << ' ' << lu << ' ' << sc.real << endl;
    }
    // this time l is combined with u reference and v reference, we manually extract them from l compared to uniGate
    [[nodiscard]] u8 getLayerIdU(u8 layer_id) const { return !l ? 0 : layer_id - 1; }
    [[nodiscard]] u8 getLayerIdV(u8 layer_id) const { return !(l & 1) ? 0 : layer_id - 1; }
};

// enum class is a better way to define enumerations in C++
// It provides better type safety and avoids name conflicts
enum class layerType
{
    INPUT,
    FFT,
    IFFT,
    ADD_BIAS,
    RELU,
    Sqr,
    OPT_AVG_POOL,
    MAX_POOL,
    AVG_POOL,
    DOT_PROD,
    PADDING,
    FCONN,
    NCONV,
    NCONV_MUL,
    NCONV_ADD
};

/**
 * layer class is a arithmetic circuit layer for ZKP
 * a circuit is composed of vectors of layers
 */
class layer
{
public:
    // layertype of this layer
    layerType ty;
    // adding {} means brace initialization to 0
    // size means number of values this layer produces
    // size_u[0] and size_v[0] means current layer subsets
    // size_u[1] and size_v[1] means previous layer subsets
    u32 size{}, size_u[2]{}, size_v[2]{};
    // bit precisions for u, v and output
    // bit_length means bit length of output values
    // bit_length_u[0] is for current layer subsets u
    // bit_length_u[1] is for previous layer subsets u
    i8 bit_length_u[2]{}, bit_length_v[2]{}, bit_length{};
    i8 max_bl_u{}, max_bl_v{};

    bool need_phase2;

    // bit decomp related
    u32 zero_start_id;

    std::vector<uniGate> uni_gates;
    std::vector<binGate> bin_gates;

    // enable to map compact subset indices to original positions
    // it stores the position of output
    vector<u32> ori_id_u, ori_id_v;
    i8 fft_bit_length;

    // iFFT or avg pooling.
    F scale;

    layer()
    {
        bit_length_u[0] = bit_length_v[0] = -1;
        size_u[0] = size_v[0] = 0;
        bit_length_u[1] = bit_length_v[1] = -1;
        size_u[1] = size_v[1] = 0;
        need_phase2 = false;
        zero_start_id = 0;
        fft_bit_length = -1;
        scale = F_ONE;
    }

    /**
     * max_bl_u and max_bl_v are the max bit lengths of u and v inputs
     * update them based on subset processing results
     */
    void updateSize()
    {
        max_bl_u = std::max(bit_length_u[0], bit_length_u[1]);
        max_bl_v = 0;
        if (!need_phase2)
            return;

        max_bl_v = std::max(bit_length_v[0], bit_length_v[1]);
    }
};

class layeredCircuit
{
public:
    vector<layer> circuit;
    // how many layers does this circuit have
    u8 size;
    vector<F> two_mul;

    void init(u8 q_bit_size, u8 _layer_sz);
    void initSubset();
};
