#include "tooth_reg.h"
#include "open3d/Open3D.h"

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include <chrono>   

int main(){
    std::string model_path = "/home/ziyang/Desktop/iMeshSegNet-Reg-ONNX/PointNetReg-sim.onnx";

    auto tl = new ToothReg(model_path);

    std::string input_mesh_path;

    while(std::cin>>input_mesh_path)
    {
        // cudaDeviceReset();
        std::vector<Eigen::Vector3d> keypoints(6);

        auto mesh = open3d::io::CreateMeshFromFile(input_mesh_path.c_str());

        auto start = std::chrono::system_clock::now();

        tl->do_infer(mesh, keypoints);

        for (auto i = 0; i < keypoints.size(); i++){
            std::cout<<keypoints[i][0]<< " ";
            std::cout<<keypoints[i][1]<< " ";
            std::cout<<keypoints[i][2]<<std::endl;
            // std::cout<<"point "<< i+1 <<std::endl;
            // std::cout<<"x:" <<keypoints[i][0]<< " ";
            // std::cout<<"y:" <<keypoints[i][1]<< " ";
            // std::cout<<"z:" <<keypoints[i][2]<<std::endl;
        }
        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout <<  "Spent " << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << " seconds." << std::endl;
        
    //     std::ofstream myfile ("example.txt");
    //     if (myfile.is_open())
    //     {
    //         for(int count = 0; count < labels.size(); count ++){
    //             myfile << result[count] << "\n" ;
    //         }
    //         myfile.close();
    //     }
    }
    return 0;
}

/**
    std::cout<<label_num<<std::endl;
    std::cout<<labels[0]<<std::endl;
    std::cout<<labels[1]<<std::endl;
    std::cout<<labels[2]<<std::endl;
    std::cout<<labels[3]<<std::endl;
    int* result = &labels[0];

    std::ofstream myfile ("example.txt");
    if (myfile.is_open())
    {
        for(int count = 0; count < labels.size(); count ++){
            myfile << result[count] << "\n" ;
        }
        myfile.close();
    }
**/