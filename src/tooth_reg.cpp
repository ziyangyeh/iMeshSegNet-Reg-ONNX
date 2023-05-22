#include "tooth_reg.h"

ToothReg::ToothReg(std::string model)
{
    model_path = model;;
}

void ToothReg::do_infer(std::shared_ptr<open3d::geometry::TriangleMesh> mesh, std::vector<Eigen::Vector3d> &results)
{
    torch::NoGradGuard no_grad;
    std::unique_ptr<Logger> logger{new Logger()};
    uint32_t flag = 1U <<static_cast<uint32_t> (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
    std::unique_ptr<InferEngine> inferRun{new InferEngine(*logger, flag, model_path)};
    auto engine = inferRun->engine;
    auto context = inferRun->context;
    std::cout<<"Engine and Context are created."<<std::endl;

    auto mwf = getTensors(mesh);

    auto batch_size = 1;
    int points_num = mwf->size;

    std::cout<<"Inference is started."<<std::endl;
    auto out_tmp{do_inference(1, points_num, mwf, context, engine)};
    std::cout<<"Inference is ended."<<std::endl;

    auto index = out_tmp.argmax(0);
    std::vector<long> index_vec(index.data_ptr<long>(), index.data_ptr<long>() + index.numel());
    std::vector<Eigen::Vector3d>keys(index_vec.size());
    #pragma omp parallel for
    for (int i = 0; i < index_vec.size(); i++)
    {
        keys[i] = mwf->barycenters.row(index_vec[i]).head(3);
    }

    auto raw = mwf->mesh;
    auto center = mwf->center;
    raw->Translate(-center);
    auto src_pcd = new open3d::geometry::PointCloud();
    src_pcd->points_ = raw->vertices_;
    auto kdtree = new open3d::geometry::KDTreeFlann(*src_pcd);

    std::vector<int> final_keys_idx (index_vec.size());

    int num = 1;
    #pragma omp parallel for
    for(int i=0; i<keys.size();i++){
        std::vector<int> new_indices_vec(num);
        std::vector<double> new_dists_vec(num);
        kdtree->SearchKNN(keys[i], num, new_indices_vec, new_dists_vec);
        final_keys_idx[i] = new_indices_vec[0];
    }

    std::vector<Eigen::Vector3d> tmp(index_vec.size());
    #pragma omp parallel for
    for(int i=0; i<keys.size();i++){
        tmp[i] = raw->vertices_[final_keys_idx[i]];
    }

    auto final_pcd = new open3d::geometry::PointCloud();
    final_pcd->points_ = tmp;
    final_pcd->Translate(center);
    #pragma omp parallel for
    for(int i=0; i<keys.size();i++){
        results[i] = final_pcd->points_[i];
    }
}

ToothReg::~ToothReg()
{
}