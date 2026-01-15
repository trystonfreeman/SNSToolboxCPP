#pragma once
#include <string>
#include <Eigen/Eigen>
#include "cjsontoEigen.h"
#include <fstream>
#include <string>
#include <iostream>

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::ArrayXXf;

namespace SNSToolboxCPP {
    class Network {
    public:
        enum class Simulator {
            SparseSim,
            DenseSim
        };
        Simulator sim = Simulator::DenseSim;

        explicit Network(const std::string& filename);
        VectorXf Forward(const VectorXf& inputs);
        void SetBackend(const std::string &backend);
        float dt;
        bool spiking;
        bool delay;
        bool elec;
        bool rect;
        bool gated;
        int num_channels;
        int num_neurons;
        int num_pop;
        int num_connections;
        int num_inputs;
        int num_outputs;
        std::string name;
        // Neuron Properties
        // (Should all be 1 by num_neurons)
        VectorXf v;
        //ArrayXXf V;
        VectorXf v_last;
        VectorXf v0;
        VectorXf vRest;
        VectorXf cM;
        VectorXf gM;
        VectorXf iB;
        VectorXf timeFactorMembrane;
        // Connection (synapses and inputs) Properties
        // (Should all be num_neurons by num_neurons)
        MatrixXf gMaxNon;
        ArrayXXf delE;
        ArrayXXf eLo;
        ArrayXXf eHi;
        MatrixXf inputConn;
        MatrixXf outConnVolt;
        // Spiking Only
        VectorXf spikes;
        VectorXf theta_0;
        VectorXf theta;
        VectorXf theta_Last;
        VectorXf m;
        VectorXf tauTheta;
        VectorXf timeFactorThreshold;
        VectorXf thetaLeak;
        VectorXf thetaIncrement;
        VectorXf thetaFloor;
        VectorXf vReset;
        ArrayXXf gMaxSpike;
        ArrayXXf gSpike;
        ArrayXXf tauSyn;
        ArrayXXf timeFactorSynapse;
        ArrayXXf outConnSpike;
        ArrayXXf gIncrement;
        // Delay Only

        // Elec Only
        ArrayXXf gElectrical;
        // Rect Only
        ArrayXXf gRectified;
        // Gated Only
        ArrayXXf gIon;
        ArrayXXf eIon;
        ArrayXXf powA;
        ArrayXXf slopeA;
        ArrayXXf kA;
        ArrayXXf eA;
        ArrayXXf powB;
        ArrayXXf slopeB;
        ArrayXXf kB;
        ArrayXXf eB;
        ArrayXXf tauMaxB;
        ArrayXXf powC;
        ArrayXXf slopeC;
        ArrayXXf kC;
        ArrayXXf eC;
        ArrayXXf tauMaxC;
        ArrayXXf bGate;
        ArrayXXf bGateLast;
        ArrayXXf bGate0;
        ArrayXXf cGate;
        ArrayXXf cGateLast;
        ArrayXXf cGate0;
        // Intermediates in Calculations I think
        MatrixXf V_last;
        VectorXf i_app;
        ArrayXXf g_syn;
        VectorXf i_syn;
        VectorXf i_gated;
        ArrayXXf V_last_gated;
        ArrayXXf a_inf;
        ArrayXXf b_inf;
        ArrayXXf c_inf;
        ArrayXXf tauB;
        ArrayXXf tauC;
        ArrayXXf i_ion;
        ArrayXXf outputs_array;
        ArrayXXf Ones;
        ArrayXXf OnesNxN;
        ArrayXXf _Zeros;

        std::vector<std::pair<int, int>> connections;
        std::vector<int> channels;
        bool CompiledSparse = false;
    private:
        void LoadFromFile(const std::string& FileName);
        void CompileSparse();
        ArrayXXf ForwardSparse(VectorXf inputs);
        ArrayXXf ForwardDense(const VectorXf &inputs);
        bool CheckInputSize(const VectorXf& inputs) const;

        static std::string readFileIntoBuffer(const std::string& filename);
    };
}