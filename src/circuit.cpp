#include "circuit.h"
#include "utils.hpp"

/**
 * This function is used in Libra to find compact combinations
 * It replace gate input indices with subset indices, like 0, 1, 2, ...
 * real position is stored
 */
void layeredCircuit::initSubset()
{
    cerr << "begin subset init." << endl;
    vector<int> visited_uidx(circuit[0].size); // whether the i-th layer, j-th gate has been visited in the current i layer
    // the subset index of the i-th layer, j-th gate
    // it includes all of the gate outputs in the current layer
    vector<u64> subset_uidx(circuit[0].size);
    vector<int> visited_vidx(circuit[0].size); // whether the i-th layer, j-th gate has been visited in the current i layer
    vector<u64> subset_vidx(circuit[0].size);  // the subset index of the i-th layer, j-th gate

    // process each layer
    for (u8 i = 1; i < size; ++i)
    {
        // cur means the current layer i, lst means the last layer i-1
        auto &cur = circuit[i], &lst = circuit[i - 1];
        // preprocess has_pre_layer_u and has_pre_layer_v first by FFT and IFFT
        bool has_pre_layer_u = circuit[i].ty == layerType::FFT || circuit[i].ty == layerType::IFFT;
        bool has_pre_layer_v = false;

        // for current layer unary gates
        for (auto &gate : cur.uni_gates)
        {
            // input u comes from previous layer->lu!=0
            // so this means it comes from current layer
            if (!gate.lu)
            {
                // if position u is not visited in layer i
                if (visited_uidx[gate.u] != i)
                {
                    // mark as visited in layer i
                    visited_uidx[gate.u] = i;
                    // maps original index to subset index
                    subset_uidx[gate.u] = cur.size_u[0];
                    // stores original input position in ori_id_u
                    cur.ori_id_u.push_back(gate.u);
                    // increment count
                    ++cur.size_u[0];
                }
                // update gate.u to remapped index
                gate.u = subset_uidx[gate.u];
            }
            has_pre_layer_u |= (gate.lu != 0);
        }

        // for current layer binary gates
        for (auto &gate : cur.bin_gates)
        {
            // current layer input u
            if (!gate.getLayerIdU(i))
            {
                // like unary gates input
                if (visited_uidx[gate.u] != i)
                {
                    visited_uidx[gate.u] = i;
                    subset_uidx[gate.u] = cur.size_u[0];
                    cur.ori_id_u.push_back(gate.u);
                    ++cur.size_u[0];
                }
                gate.u = subset_uidx[gate.u];
            }
            // current layer input v
            if (!gate.getLayerIdV(i))
            {
                if (visited_vidx[gate.v] != i)
                {
                    visited_vidx[gate.v] = i;
                    subset_vidx[gate.v] = cur.size_v[0];
                    cur.ori_id_v.push_back(gate.v);
                    ++cur.size_v[0];
                }
                gate.v = subset_vidx[gate.v];
            }
            has_pre_layer_u |= (gate.getLayerIdU(i) != 0);
            has_pre_layer_v |= (gate.getLayerIdV(i) != 0);
        }

        // up to now cur.size_u is the number of current layer gate used
        // same for cur.size_v
        // determines how many bits are needed to represent all of the outputs
        cur.bit_length_u[0] = ceilPow2BitLength(cur.size_u[0]);
        cur.bit_length_v[0] = ceilPow2BitLength(cur.size_v[0]);

        // if gates receive input from previous layers
        if (has_pre_layer_u)
            switch (cur.ty)
            {
            case layerType::FFT:
                // FFT layer array length set
                cur.size_u[1] = 1ULL << cur.fft_bit_length - 1;
                cur.bit_length_u[1] = cur.fft_bit_length - 1;
                break;
            case layerType::IFFT:
                cur.size_u[1] = 1ULL << cur.fft_bit_length;
                cur.bit_length_u[1] = cur.fft_bit_length;
                break;
            default:
                cur.size_u[1] = lst.size;
                cur.bit_length_u[1] = lst.bit_length;
                break;
            }
        // if no previous layers input, default case
        else
        {
            cur.size_u[1] = 0;
            cur.bit_length_u[1] = -1;
        }

        // do the same for v inputs
        if (has_pre_layer_v)
        {
            if (cur.ty == layerType::DOT_PROD)
            {
                cur.size_v[1] = lst.size >> cur.fft_bit_length;
                cur.bit_length_v[1] = lst.bit_length - cur.fft_bit_length;
            }
            else
            {
                cur.size_v[1] = lst.size;
                cur.bit_length_v[1] = lst.bit_length;
            }
        }
        else
        {
            cur.size_v[1] = 0;
            cur.bit_length_v[1] = -1;
        }
        cur.updateSize();
    }
    cerr << "begin subset finish." << endl;
}

/**
 * Initialize the layered circuits
 * @param q_bit_size: the max bit length of quantization, indicating the max scale of quant
 * @param _layer_sz: the number of layers in the circuit
 */
void layeredCircuit::init(u8 q_bit_size, u8 _layer_sz)
{
    // scaling factor array two_mul has many powers of 2
    // it's used for quantization and scaling between layers
    /**
     * val[i] = input_val * two_mul[scale_factor]
     * real scale factor is 2^m, we first have m then look up 2^m in the two_mul array
     * This DOESN'T mean that all circuits use one same quantization bit length
     * different layers have their specific settings, but they all look up scales from the two_mul array
     */
    two_mul.resize((q_bit_size + 1) << 1);
    // initialize base values: 1 and -1
    two_mul[0] = F_ONE;
    two_mul[q_bit_size + 1] = -F_ONE;
    // initialize other values: 2^i and -2^i
    for (int i = 1; i <= q_bit_size; ++i)
    {
        two_mul[i] = two_mul[i - 1] + two_mul[i - 1];
        two_mul[i + q_bit_size + 1] = -two_mul[i];
    }
    size = _layer_sz;
    circuit.resize(size);
}
