/**
 * @file        test_app.cpp
 * 
 * @brief       Sample application for testing features.
 * 
 * @author      Filippo Maggioli\n
 *              (maggioli@di.uniroma1.it, maggioli.filippo@gmail.com)\n
 *              Sapienza, University of Rome - Department of Computer Science
 * 
 * @date        2023-07-17
 */
#define NOMINMAX
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <rmt/rmt.hpp>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <unsupported/Eigen/SparseExtra>
#include <nlohmann/json.hpp>

#include <igl/is_edge_manifold.h>
#include <igl/is_vertex_manifold.h>
#include <igl/doublearea.h>
// namespace Eigen
// {
//     Eigen::internal::all_t all = Eigen::placeholders::all;
// };
// #include <igl/remove_unreferenced.h>
// #include <igl/remove_duplicate_vertices.h>
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <cmath>

void StartTimer();
double StopTimer();

struct rmtArgs
{
    std::string DataDir;
    std::string OutDir;
    float RemeshPctg;
    int RNG;
    bool Visualize;
};

rmtArgs ParseArgs(int argc, const char* const argv[]);
void Usage(const std::string& Prog, bool IsError = false);

int ProcessMesh(const std::filesystem::path& File, const rmtArgs& Config);

int main(int argc, const char* const argv[])
{
    auto Args = ParseArgs(argc, argv);
    double TotTime = 0.0;
    double t;

    if (Args.Visualize) {
        polyscope::init();
    }

    std::filesystem::path OutDir(Args.OutDir);
    if (!std::filesystem::exists(OutDir)) {
        std::filesystem::create_directories(OutDir);
    }

    std::filesystem::path DataDir(Args.DataDir);
    for (auto const& Content : std::filesystem::directory_iterator{DataDir}) {

        if (!std::filesystem::is_regular_file(Content)) continue;

        t = 0.0f;
        StartTimer();
        if (ProcessMesh(Content, Args) != 0) {
            return 1;
        }
        t = StopTimer();
        TotTime += t;

    }

    std::cout << "Program terminated successfully in " << TotTime << "s." << std::endl;
    return 0;
}



int ProcessMesh(const std::filesystem::path& File, const rmtArgs& Config) {

    std::string FileStr = File.string();
    std::cout << "Processing mesh " << FileStr << "... " << std::endl;
    rmt::Mesh Mesh(FileStr);

    size_t NVOrig = Mesh.NumVertices();
    size_t NTOrig = Mesh.NumTriangles();

    std::cout << "Repairing non manifoldness... " << std::endl;
    Mesh.MakeManifold();
    size_t NVManif = Mesh.NumVertices();
    size_t NTManif = Mesh.NumTriangles();

    std::cout << "Normalizing mesh... " << std::endl;
    Eigen::VectorXd FaceAreas;
    igl::doublearea(Mesh.GetVertices(), Mesh.GetTriangles(), FaceAreas);
    Mesh.CenterAtOrigin();
    double Area = (FaceAreas.array() / 2).sum();
    Mesh.Scale(1.0 / std::sqrt(Area));
    
    std::cout << "Computing mesh edges and boundaries... " << std::endl;
    Mesh.ComputeEdgesAndBoundaries();

    std::cout << "Computing Voronoi FPS with density " << Config.RemeshPctg << "..." << std::endl;
    size_t NVRemesh = std::floor(Config.RemeshPctg * NVOrig);
    rmt::VoronoiPartitioning VPart(Mesh, Config.RNG);
    while (VPart.NumSamples() < NVRemesh)
        VPart.AddSample(VPart.FarthestVertex());

    std::cout << "Refining sampling to ensure closed ball property... " << std::endl;
    rmt::FlatUnion FU(Mesh, VPart);
    do
    {
        FU.DetermineRegions();
        FU.ComputeTopologies();
    } while (!FU.FixIssues());
    size_t NVRefined = VPart.NumSamples();
    
    /* Compute cluster index for each high-res vertex */
    Eigen::VectorXi PartVec = VPart.GetPartitions();
    Eigen::MatrixXd FeatVec = PartVec.cast<double>();
    
    if (Config.Visualize) {
        polyscope::registerSurfaceMesh("Original Mesh", Mesh.GetVertices(), Mesh.GetTriangles());
        polyscope::getSurfaceMesh("Original Mesh")->addVertexScalarQuantity("Voronoi Regions", PartVec.cast<double>());
        polyscope::show();
    }

    std::cout << "Reconstructing mesh... " << std::endl;
    Eigen::MatrixXd VV;
    Eigen::MatrixXi FF;
    rmt::MeshFromVoronoi(Mesh.GetVertices(), Mesh.GetTriangles(), VPart, VV, FF);
    rmt::CleanUp(VV, FF);
    size_t NVFinal = VV.rows();
    size_t NTFinal = FF.rows();

    std::cout << "== Current mesh statistics ==" << std::endl;
    std::cout << "\t(Original mesh) Vertex count: " << NVOrig << " -- Triangle count: " << NTOrig << std::endl;
    std::cout << "\t(Manifold repair) Vertex count: " << NVManif << " -- Triangle count: " << NTManif << std::endl;
    std::cout << "\t(Voronoi sampling) Base sample size: " << NVRemesh << " -- Refined sample: " << NVRefined << std::endl;
    std::cout << "\t(Remeshing) Vertex count: " << NVFinal << " -- Triangle count: " << NTFinal << std::endl;
    if (FF.rows() == 0)
    {
        std::cerr << "WARNING: Remesh has zero triangles. Maybe there are too many connected components?" << std::endl;
    }

    std::cout << "Exporting... " << std::endl;
    std::string FileName = File.stem().string();
    std::filesystem::path OutDir(Config.OutDir);
    std::filesystem::path RemeshOutFile = OutDir / (FileName + "_remesh.ply");
    std::filesystem::path VoronoiOutFile = OutDir / (FileName + "_voronoi.ply");
    if (!rmt::ExportMesh(RemeshOutFile.string(), VV, FF))
    {
        std::cerr << "Cannot write mesh." << std::endl;
        return -1;
    }
    if (!rmt::ExportMesh(VoronoiOutFile.string(), Mesh.GetVertices(), Mesh.GetTriangles(), FeatVec))
    {
        std::cerr << "Cannot write mesh." << std::endl;
        return -1;
    }

    return 0;
}


std::chrono::system_clock::time_point Start;
void StartTimer()
{
    Start = std::chrono::system_clock::now();
}

double StopTimer()
{
    std::chrono::system_clock::time_point End;
    End = std::chrono::system_clock::now();
    std::chrono::system_clock::duration ETA;
    ETA = End - Start;
    size_t ms;
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(ETA).count();
    return ms * 1.0e-3;
}

rmtArgs ParseArgs(int argc, const char* const argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        std::string argvi(argv[i]);
        if (argvi == "-h" || argvi == "--help")
        {
            Usage(argv[0]);
            exit(0);
        }
    }

    rmtArgs Args;
    Args.DataDir = "";
    Args.OutDir = "";
    Args.RemeshPctg = 1.0;
    Args.Visualize = false;
    Args.RNG = 0;

    for (int i = 1; i < argc; ++i)
    {
        std::string argvi(argv[i]);
        // if (argvi == "-f" || argvi == "--file")
        // {
        //     if (i == argc - 1)
        //     {
        //         Usage(argv[0], true);
        //     }
        //     return ParseFromFile(argv[i + 1]);
        // }
        if (argvi == "-o" || argvi == "--output")
        {
            if (i == argc - 1)
            {
                Usage(argv[0], true);
            }
            Args.OutDir = argv[++i];
            continue;
        }
        if (argvi == "-s" || argvi == "--seed")
        {
            if (i == argc - 1)
            {
                Usage(argv[0], true);
            }
            Args.RNG = std::stoi(argv[++i]);
            continue;
        }
        if (argvi == "-v" || argvi == "--visual")
        {
            Args.Visualize = true;
            continue;
        }
        if (Args.DataDir.empty())
            Args.DataDir = argvi;
        else
            Args.RemeshPctg = std::stof(argvi);
    }

    if (Args.DataDir.empty())
    {
        std::cerr << "No data directory given." << std::endl;
        Usage(argv[0], true);
    }
    if (Args.RemeshPctg == -1)
    {
        std::cerr << "No remeshing density percentage given." << std::endl;
        Usage(argv[0], true);
    }
    if (Args.OutDir.empty())
    {
        std::filesystem::path OutDir = std::filesystem::path(Args.DataDir) / "decomp";
        Args.OutDir = OutDir.string();
    }

    return Args;
}

void Usage(const std::string& Prog, bool IsError)
{
    std::ostream* _out = &std::cout;
    if (IsError)
        _out = &std::cerr;
    std::ostream& out = *_out;

    out << std::endl;
    out << Prog << " usage:" << std::endl;
    out << std::endl;
    out << "\t" << Prog << " data_dir remesh_pctg [-o|--output out_dir] [-s|--seed rng] [-v|--visual]" << std::endl;
    out << "\t" << Prog << " -f|--file config_file" << std::endl;
    out << "\t" << Prog << " -h|--help" << std::endl;
    out << std::endl;
    out << "Arguments details:" << std::endl;
    out << "\t- data_dir is the directory containing the mesh dataset;" << std::endl;
    out << "\t- remesh_pctg is the fraction of input vertices you want in the remesh;" << std::endl;
    out << "\t- -o|--output sets the output directory for the processed dataset;" << std::endl;
    out << "\t- -s|--seed sets the seed for random generation (used for selecting remesh vertices);" << std::endl;
    out << "\t- -v|--visual if provided, the script will show Voronoi decompositions via Polyscope as it runs;" << std::endl;
    out << "\t- -f|--file sets the arguments using the content of config_file." << std::endl;
    out << "\t- -h|--help prints this message." << std::endl;

    if (IsError)
        exit(-1);
}


// rmtArgs ParseFromFile(const std::string& Filename)
// {
//     std::ifstream Stream;
//     Stream.open(Filename, std::ios::in);
//     if (!Stream.is_open())
//     {
//         std::cerr << "Cannot open file " << Filename << " for reading." << std::endl;
//         exit(-1);
//     }

//     nlohmann::json j;
//     try
//     {
//         Stream >> j;
//     }
//     catch(const std::exception& e)
//     {
//         std::cerr << e.what() << '\n';
//         exit(-1);
//     }
    
    
//     Stream.close();

//     if (!j.contains("input_mesh"))
//     {
//         std::cerr << "Configuration file must contain the \'input_mesh\' attribute." << std::endl;
//         exit(-1);
//     }
//     if (!j.contains("num_samples"))
//     {
//         std::cerr << "Configuration file must contain the \'num_samples\' attribute." << std::endl;
//         exit(-1);
//     }

//     if (!j["input_mesh"].is_string())
//     {
//         std::cerr << "\'input_mesh\' attribute must be a string." << std::endl;
//         exit(-1);
//     }
//     if (!j["num_samples"].is_number_integer())
//     {
//         std::cerr << "\'num_samples\' attribute must be an integer numeric value." << std::endl;
//         exit(-1);
//     }

//     rmtArgs Args;
//     Args.InMesh = j["input_mesh"];
//     Args.NumSamples = j["num_samples"];
//     Args.Resampling = false;
//     Args.Evaluate = false;
//     Args.OutMesh = std::filesystem::path(Args.InMesh).filename().string();
//     Args.OutMesh = (std::filesystem::current_path() / std::filesystem::path(Args.OutMesh)).string();

//     if (j.contains("resampling"))
//     {
//         if (!j["resampling"].is_boolean())
//         {
//             std::cerr << "When provided, \'resampling\' attribute must be boolean." << std::endl;
//             exit(-1);
//         }
//         Args.Resampling = j["resampling"];
//     }

//     if (j.contains("evaluate"))
//     {
//         if (!j["evaluate"].is_boolean())
//         {
//             std::cerr << "When provided, \'evaluate\' attribute must be boolean." << std::endl;
//             exit(-1);
//         }
//         Args.Evaluate = j["evaluate"];
//     }

//     if (j.contains("out_mesh"))
//     {
//         if (!j["out_mesh"].is_string())
//         {
//             std::cerr << "When provided, \'out_mesh\' attribute must be a string." << std::endl;
//             exit(-1);
//         }
//         Args.OutMesh = j["out_mesh"];
//     }

//     return Args;
// }