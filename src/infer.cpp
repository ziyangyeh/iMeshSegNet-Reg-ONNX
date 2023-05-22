#include "infer.h"

torch::Tensor EigenMatrixToTorchTensor(Eigen::MatrixXd e)
{
    auto t = torch::rand({e.cols(), e.rows()}, torch::TensorOptions().requires_grad(false));
    float *data = t.data_ptr<float>();

    Eigen::Map<Eigen::MatrixXf> ef(data, t.size(1), t.size(0));
    ef = e.cast<float>();
    // t.requires_grad_(false);
    return t.transpose(0, 1);
}

std::shared_ptr<MeshWithFeature> getTensors(std::shared_ptr<open3d::geometry::TriangleMesh> origin_mesh)
{
    auto origin_center = origin_mesh->GetCenter();
    origin_mesh->Translate(-origin_mesh->GetCenter());
    auto mesh = origin_mesh->SimplifyQuadricDecimation(3000, std::numeric_limits<double>::infinity(), 1.0);
    mesh->ComputeTriangleNormals(true);

    auto tmp_pts = mesh->vertices_;

    auto tmp_tri = mesh->triangles_;

    auto tmp_nor = mesh->triangle_normals_;

    Eigen::MatrixXd cells = Eigen::MatrixXd::Zero((int)tmp_tri.size(), 9);
    Eigen::MatrixX3d barycenters = Eigen::MatrixX3d::Zero((int)tmp_tri.size(), 3);
#pragma omp parallel for
    for (auto &iter : tmp_tri)
    {
        auto index = &iter - &tmp_tri[0];
        Eigen::Matrix<double, 1, 9> sub_cord;
        sub_cord[0] = tmp_pts.at(iter[0])[0];
        sub_cord[1] = tmp_pts.at(iter[0])[1];
        sub_cord[2] = tmp_pts.at(iter[0])[2];
        sub_cord[3] = tmp_pts.at(iter[1])[0];
        sub_cord[4] = tmp_pts.at(iter[1])[1];
        sub_cord[5] = tmp_pts.at(iter[1])[2];
        sub_cord[6] = tmp_pts.at(iter[2])[0];
        sub_cord[7] = tmp_pts.at(iter[2])[1];
        sub_cord[8] = tmp_pts.at(iter[2])[2];

        Eigen::Matrix<double, 1, 3> sub_cent;
        sub_cent[0] = (sub_cord[0] + sub_cord[3] + sub_cord[6]) / 3;
        sub_cent[1] = (sub_cord[1] + sub_cord[4] + sub_cord[7]) / 3;
        sub_cent[2] = (sub_cord[2] + sub_cord[5] + sub_cord[8]) / 3;

        cells.row(index) = sub_cord;
        barycenters.row(index) = sub_cent;
    }
    auto decimated_barycenters = barycenters;

    Eigen::MatrixX3d mesh_normals = Eigen::MatrixX3d::Zero((int)tmp_nor.size(), 3);
#pragma omp parallel for
    for (auto &iter : tmp_nor)
    {
        mesh_normals.row(&iter - &tmp_nor[0]) = tmp_nor[&iter - &tmp_nor[0]];
    }

    Eigen::MatrixX3d points = Eigen::MatrixX3d::Zero((int)tmp_pts.size(), 3);
#pragma omp parallel for
    for (auto &iter : tmp_pts)
    {
        points.row(&iter - &tmp_pts[0]) = tmp_pts[&iter - &tmp_pts[0]];
    }

    auto maxs = points.colwise().maxCoeff();
    auto mins = points.colwise().minCoeff();
    auto means = points.colwise().mean();
    Eigen::Array3d stds;
#pragma omp parallel for
    for (int i = 0; i < stds.size(); i++)
    {
        stds[i] = sqrt(((points.col(i).array() - points.col(i).mean()).square().sum() / (points.col(i).size() - 1)));
    }

    auto normals = mesh_normals;
    auto nor_means = mesh_normals.colwise().mean();
    Eigen::Array3d nor_stds;
#pragma omp parallel for
    for (int i = 0; i < nor_stds.size(); i++)
    {
        nor_stds[i] = sqrt(((mesh_normals.col(i).array() - mesh_normals.col(i).mean()).square().sum() / (mesh_normals.col(i).size() - 1)));
    }

#pragma omp parallel for
    for (int i = 0; i < 3; i++)
    {
        cells.col(i) = (cells.col(i).array() - means[i]) / stds[i];
        cells.col(i + 3) = (cells.col(i + 3).array() - means[i]) / stds[i];
        cells.col(i + 6) = (cells.col(i + 6).array() - means[i]) / stds[i];
        barycenters.col(i) = (barycenters.col(i).array() - mins[i]) / (maxs[i] - mins[i]);
        normals.col(i) = (normals.col(i).array() - nor_means[i]) / nor_stds[i];
    }

    Eigen::MatrixXd X = Eigen::MatrixXd::Zero((int)tmp_tri.size(), cells.cols() + barycenters.cols() + mesh_normals.cols());
    X << cells, barycenters, normals;

    std::vector<at::Tensor> output_tensor;
    output_tensor.push_back(EigenMatrixToTorchTensor(X).transpose(0, 1).ravel());

    origin_mesh->Translate(origin_center);
    std::shared_ptr<MeshWithFeature> mwf{new MeshWithFeature};
    mwf->mesh = origin_mesh;
    mwf->center = origin_center;
    mwf->tensors = output_tensor;
    mwf->barycenters = decimated_barycenters;
    mwf->size = tmp_tri.size();
    return mwf;
}

torch::Tensor do_inference(int batch_size, int points_num, std::shared_ptr<MeshWithFeature> meshwithfeature, std::shared_ptr<nvinfer1::IExecutionContext> context, std::shared_ptr<nvinfer1::ICudaEngine> engine)
{
    context->setInputShape("input", nvinfer1::Dims3(batch_size, 15, points_num));

    auto output = torch::zeros({batch_size, points_num, 6}, torch::TensorOptions().requires_grad(false)).ravel();

    const int inputIndex = engine->getBindingIndex("input");
    const int outputIndex = engine->getBindingIndex("output");

    void *buffers[2];

    auto X = meshwithfeature->tensors[0];

    cudaMalloc(&buffers[inputIndex], X.numel() * sizeof(float));
    cudaMalloc(&buffers[outputIndex], output.numel() * sizeof(float));

    std::vector<float> input_v(X.data_ptr<float>(), X.data_ptr<float>() + X.numel());
    float *input_f = &input_v[0];
    std::vector<float> output_v(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
    float *output_f = &output_v[0];

    cudaMemcpy(buffers[inputIndex], input_f, X.numel() * sizeof(float), cudaMemcpyHostToDevice);
    context->executeV2(buffers);
    cudaMemcpy(output_f, buffers[outputIndex], output.numel() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);

    return torch::from_blob(output_f, {batch_size, points_num, 6}, torch::TensorOptions().requires_grad(false)).squeeze({0});
}
