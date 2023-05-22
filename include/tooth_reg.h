#ifndef __TOOTH_REG_H__
#define __TOOTH_REG_H__

#include "engine.h"
#include "infer.h"

class ToothReg
{
private:
    std::string model_path;
public:
    ToothReg(std::string);
    void do_infer(std::shared_ptr<open3d::geometry::TriangleMesh>, std::vector<Eigen::Vector3d> &results);
    ~ToothReg();
};

#endif