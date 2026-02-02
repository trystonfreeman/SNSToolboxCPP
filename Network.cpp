#include "include/Network.h"
namespace SNSToolboxCPP{
    Network::Network(const std::string& filename)
    {
        LoadFromFile(filename);
    }

    void Network::SetBackend(const std::string &backend) {
        if (backend == "DenseSim") {
            sim = Simulator::DenseSim;
        }
        else if (backend == "SparseSim") {

            sim = Simulator::SparseSim;
        }
        else {
            std::cerr<<"ERROR: Backend \""<<backend<<"\" not supported."<<std::endl;
        }
    }

    VectorXf Network::Forward(const VectorXf& inputs) {
        auto outputs = VectorXf(num_outputs);
        if (!CheckInputSize(inputs)) {
            std::cerr <<"ERROR: Input should be vector of length "<<num_inputs<<std::endl;
            return outputs;
        }
        switch(sim) {
            case Simulator::DenseSim:
                outputs = ForwardDense(inputs);
            break;
            case Simulator::SparseSim:
                
                outputs = ForwardSparse(inputs);
            break;

        }


        return outputs;
    }

    bool Network::CheckInputSize(const VectorXf& inputs) const {
        if (inputs.cols() == num_inputs) {
            return true;
        }
        else {
            return false;
        }
    }
    void Network::LoadFromFile(const std::string& FileName) {
        const std::string buf = readFileIntoBuffer(FileName);
        cJSON *json = cJSON_Parse(buf.c_str());

        dt = cjsontoEigen::cjsontodouble(json,"dt");
        spiking = cjsontoEigen::cjsontobool(json,"spiking");
        delay = cjsontoEigen::cjsontobool(json,"delay");
        elec = cjsontoEigen::cjsontobool(json,"elec");
        rect = cjsontoEigen::cjsontobool(json,"rect");
        gated = cjsontoEigen::cjsontobool(json,"gated");
        num_channels = cjsontoEigen::cjsontoint(json,"numChannels");
        num_neurons = cjsontoEigen::cjsontoint(json,"numNeurons");
        num_pop = cjsontoEigen::cjsontoint(json,"numPop");
        num_connections = cjsontoEigen::cjsontoint(json,"numConn");
        num_inputs = cjsontoEigen::cjsontoint(json,"numInputs");
        num_outputs = cjsontoEigen::cjsontoint(json,"numOutputs");
        v = cjsontoEigen::cjsontoVectorXf(json,"v");
        v_last = cjsontoEigen::cjsontoVectorXf(json,"vLast");
        v0 = cjsontoEigen::cjsontoVectorXf(json,"v0");
        vRest = cjsontoEigen::cjsontoVectorXf(json,"vRest");
        cM = cjsontoEigen::cjsontoVectorXf(json,"cM");
        gM = cjsontoEigen::cjsontoVectorXf(json,"gM");
        iB = cjsontoEigen::cjsontoVectorXf(json,"iB");
        timeFactorMembrane = cjsontoEigen::cjsontoVectorXf(json,"timeFactorMembrane");
        gMaxNon = cjsontoEigen::cjsontoArrayXXf(json,"gMaxNon");
        delE = cjsontoEigen::cjsontoArrayXXf(json,"delE");
        eLo = cjsontoEigen::cjsontoArrayXXf(json,"eLo");
        eHi = cjsontoEigen::cjsontoArrayXXf(json,"eHi");
        inputConn = cjsontoEigen::cjsontoArrayXXf(json,"inputConn");
        outConnVolt = cjsontoEigen::cjsontoArrayXXf(json,"outConnVolt");
        if (spiking) {
            spikes = cjsontoEigen::cjsontoVectorXf(json,"spikes");
            theta_0 = cjsontoEigen::cjsontoVectorXf(json,"theta_0");
            theta = cjsontoEigen::cjsontoVectorXf(json,"theta");
            theta_Last = cjsontoEigen::cjsontoVectorXf(json,"theta_Last");
            m = cjsontoEigen::cjsontoVectorXf(json,"m");
            tauTheta = cjsontoEigen::cjsontoVectorXf(json,"tauTheta");
            timeFactorThreshold = cjsontoEigen::cjsontoVectorXf(json,"timeFactorThreshold");
            thetaLeak = cjsontoEigen::cjsontoVectorXf(json,"thetaLeak");
            thetaIncrement = cjsontoEigen::cjsontoVectorXf(json,"thetaIncrement");
            thetaFloor = cjsontoEigen::cjsontoVectorXf(json,"thetaFloor");
            vReset = cjsontoEigen::cjsontoVectorXf(json,"vReset");
            gMaxSpike = cjsontoEigen::cjsontoArrayXXf(json,"gMaxSpike");
            gSpike = cjsontoEigen::cjsontoArrayXXf(json,"gSpike");
            tauSyn = cjsontoEigen::cjsontoVectorXf(json,"tauSyn");
            timeFactorSynapse = cjsontoEigen::cjsontoVectorXf(json,"timeFactorSynapse");
            outConnSpike = cjsontoEigen::cjsontoVectorXf(json,"outConnSpike");
            gIncrement = cjsontoEigen::cjsontoVectorXf(json,"gIncrement");
        }
        if (elec) {
            gElectrical = cjsontoEigen::cjsontoArrayXXf(json,"gElectrical");
        }
        if (rect) {
            gRectified = cjsontoEigen::cjsontoArrayXXf(json,"gRectified");
        }
        if (gated) {
            gIon = cjsontoEigen::cjsontoArrayXXf(json,"gIon");
            eIon = cjsontoEigen::cjsontoArrayXXf(json,"eIon");
            powA = cjsontoEigen::cjsontoArrayXXf(json,"powA");
            powB = cjsontoEigen::cjsontoArrayXXf(json,"powB");
            powC = cjsontoEigen::cjsontoArrayXXf(json,"powC");
            slopeA = cjsontoEigen::cjsontoArrayXXf(json,"slopeA");
            slopeB = cjsontoEigen::cjsontoArrayXXf(json,"slopeB");
            slopeC = cjsontoEigen::cjsontoArrayXXf(json,"slopeC");
            kA = cjsontoEigen::cjsontoArrayXXf(json,"kA");
            kB = cjsontoEigen::cjsontoArrayXXf(json,"kB");
            kC = cjsontoEigen::cjsontoArrayXXf(json,"kC");
            eA = cjsontoEigen::cjsontoArrayXXf(json,"eA");
            eB = cjsontoEigen::cjsontoArrayXXf(json,"eB");
            eC = cjsontoEigen::cjsontoArrayXXf(json,"eC");
            tauMaxB = cjsontoEigen::cjsontoArrayXXf(json,"tauMaxB");
            tauMaxC = cjsontoEigen::cjsontoArrayXXf(json,"tauMaxC");
            bGate = cjsontoEigen::cjsontoArrayXXf(json,"bGate");
            cGate = cjsontoEigen::cjsontoArrayXXf(json,"cGate");
            bGate0 = cjsontoEigen::cjsontoArrayXXf(json,"bGate0");
            cGate0 = cjsontoEigen::cjsontoArrayXXf(json,"cGate0");
            bGateLast = cjsontoEigen::cjsontoArrayXXf(json,"bGateLast");
            cGateLast = cjsontoEigen::cjsontoArrayXXf(json,"cGateLast");
        }
    }
    std::string Network::readFileIntoBuffer(const std::string& filename) {
        // Open the file in binary mode
        std::ifstream file(filename, std::ios::binary);

        // Check if the file opened successfully
        if (!file) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return ""; // Return an empty buffer or handle error as appropriate
        }

        // Use iterators to read the entire file content into a string
        // The default-constructed iterator acts as the "end-of-stream" indicator
        std::string buffer((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());

        // The file is automatically closed when the std::ifstream object goes out of scope (RAII).
        return buffer;
    }
    void Network::CompileSparse() {
        int current_connection = 0;
        int current_spiking_connection = 0;
        int current_spiking_neuron = 0;
        for (int i = 0; i < num_neurons; i++) {
            for (int j = 0; j < num_neurons; j++) {
                if (gMaxNon(i,j) != 0) {
                    connections[current_connection] = { i,j };
                    current_connection++;
                }
                if (gMaxSpike(i,j) != 0)
                {
                    spiking_connections[current_spiking_connection] = { i,j };
                    current_spiking_connection++;
                }
            }
            for (int j = 0; j < num_channels; j++) {
                if (gIon(j, i) != 0) {
                    channels[j] = i;

                }
            }
            if (m(i) != 0)
            {
                spiking_neurons[current_spiking_neuron] = i;
                current_spiking_neuron++;
            }
        }
        CompiledSparse = true;
    }
    ArrayXXf Network::ForwardSparse(const VectorXf& inputs) {
        std::cerr << "Sparse backend not implemented yet"<< std::endl;
        if (!CompiledSparse) {
            CompileSparse();
        }
        // Implementation Here
        v_last = v;
        i_app = inputConn * inputs;
        i_syn = VectorXf::Zero(num_neurons);
        for (const auto& [sending_nrn,recieving_nrn] : connections)
        {
            const float g_syn_nrn = std::max(0.0f,std::min(gMaxNon(sending_nrn,recieving_nrn)*(v(recieving_nrn)-eLo(sending_nrn,recieving_nrn))/(eHi(sending_nrn,recieving_nrn)-eLo(sending_nrn,recieving_nrn)),gMaxNon(sending_nrn,recieving_nrn)));

            i_syn(sending_nrn) = i_syn(sending_nrn) + g_syn_nrn*(delE(sending_nrn,recieving_nrn)-v(sending_nrn));
        }
        for (const auto& [sending_nrn,recieving_nrn] : spiking_connections)
        {

        }
        for (int j = 0; j < num_channels; j++)
        {

            a_inf(j,channels[j]) = 1/(1 + kA(j,channels[j]) * std::exp(slopeA(j,channels[j]) * (eA(j,channels[j]) - v_last(channels[j]))));
            b_inf(j,channels[j]) = 1/(1 + kB(j,channels[j]) * std::exp(slopeB(j,channels[j]) * (eB(j,channels[j]) - v_last(channels[j]))));
            c_inf(j,channels[j]) = 1/(1 + kC(j,channels[j]) * std::exp(slopeC(j,channels[j]) * (eC(j,channels[j]) - v_last(channels[j]))));


            tauB(j,channels[j]) = tauMaxB(j,channels[j]) * b_inf(j,channels[j]) * sqrt(kB(j,channels[j]) * std::exp(slopeB(j,channels[j])*(eB(j,channels[j]) - v_last(channels[j]))));
            tauC(j,channels[j]) = tauMaxC(j,channels[j]) * c_inf(j,channels[j]) * sqrt(kC(j,channels[j]) * std::exp(slopeC(j,channels[j])*(eC(j,channels[j]) - v_last(channels[j]))));

            bGateLast(j,channels[j]) = bGate(j,channels[j]);
            cGateLast(j,channels[j]) = cGate(j,channels[j]);

            bGate(j,channels[j]) = bGateLast(j,channels[j]) + dt * ((b_inf(j,channels[j]) - bGateLast(j,channels[j])) / tauB(j,channels[j]));
            cGate(j,channels[j]) = cGateLast(j,channels[j]) + dt * ((c_inf(j,channels[j]) - cGateLast(j,channels[j])) / tauC(j,channels[j]));
            // i_ion = gIon * Eigen::pow(a_inf, powA) * Eigen::pow(bGate, powB) * Eigen::pow(cGate, powC) * (eIon - V_last_gated);
            i_ion(j,channels[j]) = gIon(j,channels[j]) * (eIon(j,channels[j]) - v_last(channels[j])) *
                    pow(a_inf(j,channels[j]),powA(j,channels[j]))*
                    pow(bGate(j,channels[j]),powB(j,channels[j]))*
                    pow(cGate(j,channels[j]),powC(j,channels[j]));

            i_gated(channels[j]) += i_ion(j,channels[j]);
        }
        v = v_last + (timeFactorMembrane.array() * (-gM.array() * (v_last - vRest).array() + iB.array() + i_syn.array() + i_app.array() + i_gated.array())).matrix();
        ArrayXXf Outputs = (outConnVolt * v.matrix()).array();
        return Outputs;
    }

    ArrayXXf Network::ForwardDense(const VectorXf &inputs) {
        v_last = v;
        V_last = Ones.matrix() * v_last.transpose();
        i_app = inputConn * inputs;
        g_syn = _Zeros.max(gMaxNon.array().min(gMaxNon.array() * (V_last.array() - eLo) / (eHi - eLo)));
        if (spiking) {
            theta_Last = theta;
            gSpike = gSpike * (OnesNxN - timeFactorSynapse);
            g_syn = g_syn + gSpike;
        }
        i_syn = (g_syn * (delE - V_last.transpose().array())).matrix() * Ones.matrix();
        if (elec){
            std::cerr << "Electrical Synapse not supported yet" << std::endl;
            //i_syn = i_syn + ((gElectrical*V_last.array()).matrix() *  MatrixXf::Ones(num_neurons,1)).array();
        }
        //if (rect){
            std::cerr <<"Electrical Synapse not supported yet" << std::endl;
            //MatrixXf V_last = MatrixXf::Ones(num_neurons,1)*v_last.matrix();
            //ArrayXXf mask = (((V_last.transpose() - V_last).array()) > ArrayXXf::Zero(num_neurons,num_neurons)).cast<float>()*2.f-1.f;
            //ArrayXXf masked_g = mask*gRectified;

        //}
        i_gated = VectorXf::Zero(num_neurons);
        if (gated) {
            // Probably faster to create some placeholder variables for repeated expressions
            V_last_gated = (v_last * Eigen::MatrixXf::Ones(1, num_channels)).array();
            a_inf = (ArrayXXf::Ones(num_neurons, num_channels) + kA * (slopeA * (eA - V_last_gated)).exp()).pow(-1);
            b_inf = (ArrayXXf::Ones(num_neurons, num_channels) + kB * (slopeB * (eB - V_last_gated)).exp()).pow(-1);
            c_inf = (ArrayXXf::Ones(num_neurons, num_channels) + kC * (slopeC * (eC - V_last_gated)).exp()).pow(-1);
            tauB = tauMaxB * b_inf * (kB * Eigen::exp(slopeB * (eB - V_last_gated))).sqrt();
            tauC = tauMaxC * c_inf * (kC * Eigen::exp(slopeC * (eC - V_last_gated))).sqrt();
            bGateLast = bGate;
            cGateLast = cGate;
            bGate = bGateLast + dt * ((b_inf - bGateLast) / tauB);
            cGate = cGateLast + dt * ((c_inf - cGateLast) / tauC);
            i_ion = gIon * Eigen::pow(a_inf, powA) * Eigen::pow(bGate, powB) * Eigen::pow(cGate, powC) * (eIon - V_last_gated);
            i_gated = i_ion.matrix() * VectorXf::Ones(num_channels);
        }
        v = v_last + (timeFactorMembrane.array() * (-gM.array() * (v_last - vRest).array() + iB.array() + i_syn.array() + i_app.array() + i_gated.array())).matrix();
        ArrayXXf Outputs = (outConnVolt * v.matrix()).array();
        return Outputs;
    }
}
