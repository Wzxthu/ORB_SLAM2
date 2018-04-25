/**
* This file is part of CNN-SLAM.
*
* [Copyright of CNN-SLAM]
* Copyright (C) 2018
* Kai Yu <kaiy1 at andrew dot cmu dot edu> (Carnegie Mellon University)
* Zhongxu Wang <zhongxuw at andrew dot cmu dot edu> (Carnegie Mellon University)
* Manchen Wang <manchen2 at andrew dot cmu dot edu> (Carnegie Mellon University)
* For more information see <https://github.com/raulmur/CNN_SLAM>
*
* CNN-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* CNN-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with CNN-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include <opencv2/core/core.hpp>
#include <cstring>
#include <opencv2/highgui/highgui.hpp>
#include <DepthEstimation/DepthEstimator.h>
#include <numpy/arrayobject.h>
#include <python2.7/tupleobject.h>

#include <thread>
#include <mutex>

using namespace std;

namespace cnn_slam {
    DepthEstimator::DepthEstimator() : mInitialized(false) {
        mHeight = 228;
        mWidth = 304;
        mTrainingFocalLength = sqrt(powf(5.1885790117450188e+02f, 2) + powf(5.1946961112127485e+02f, 2));
        mDepthRatio = sqrt(powf(5.8262448167737955e+02f, 2) + powf(5.8269103270988637e+02f, 2));
        mpModule = nullptr;
        mpFunc = nullptr;
        mModelPath = "./FCRN-DepthPrediction/NYU_FCRN.ckpt";
    }

    void DepthEstimator::Initialize() {
        if (mInitialized)
            return;

        thread init_thread([this]() {
            cout << "Initializing depth estimator asynchronously..." << endl;

            Py_Initialize();
            import_array();
            if (!Py_IsInitialized()) {
                return;
            }

            std::string chdir_cmd = std::string("sys.path.append('./FCRN-DepthPrediction')");
            const char *cstr_cmd = chdir_cmd.c_str();
            PyRun_SimpleString("import sys");
            PyRun_SimpleString(cstr_cmd);
            PyRun_SimpleString("import numpy as np");
            PyRun_SimpleString("import tensorflow as tf");
            PyRun_SimpleString("from PIL import Image");
            PyRun_SimpleString("import models");
            PyRun_SimpleString("import os");

            // import the module
            char *modulePath = "predict";
            mpModule = PyImport_ImportModule(modulePath);
            if (!mpModule) {
                cerr << "Cannot import module \"predict\"!" << endl;
                exit(-1);
            }

            PyObject *pClass = PyObject_GetAttrString(mpModule, "Predict");
            if (!pClass) {
                cerr << "Cannot import class \"Predict\"" << endl;
                exit(-1);
            }
            PyObject *init_arg = PyString_FromString(mModelPath);
            PyObject *args = PyTuple_Pack(1, init_arg);
            mpInstance = PyInstance_New(pClass, args, nullptr);
            if (!mpInstance) {
                cerr << "Cannot create instance!" << endl;
                exit(-1);
            }

            mInitialized = true;

            cout << "Depth estimator initialized!" << endl;
        });
        init_thread.detach();
    }

    void DepthEstimator::EstimateDepth(const cv::Mat& im,
                                       cv::Mat& depth,
                                       float focalLength) {
        while (!mInitialized)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        {
            std::unique_lock<mutex> lock(mForwardMutex);

//            cout << "Starting depth prediction!" << endl << flush;

            int ori_rows = im.rows;
            int ori_cols = im.cols;
            cv::Mat input;
            resize(im, input, cv::Size(mWidth, mHeight), 0, 0, CV_INTER_LINEAR);
            unsigned char *data = input.data;
            npy_intp Dims[3] = {input.rows, input.cols, 3};
            PyObject *PyArray = PyArray_SimpleNewFromData(3, Dims, NPY_UBYTE, data);
            PyObject *pParm = PyTuple_Pack(1, PyArray);

            PyObject *func_name = PyString_FromString("depth_predict");

//            cout << "Forwarding network..." << endl;
            PyObject *pRetVal = PyObject_CallMethodObjArgs(mpInstance, func_name, pParm, NULL);
            if (!pRetVal) {
                cerr << "Error during calling depth estimation function!" << endl;
                exit(-1);
            }

            // parse the return value into a opencv mat
            PyArrayObject *Pynp_ret = (PyArrayObject *) PyArray_FromAny(pRetVal, PyArray_DescrFromType(NPY_FLOAT32), 2,
                                                                        2,
                                                                        NPY_ARRAY_CARRAY, NULL);
            cv::Mat pred(PyArray_DIM(Pynp_ret, 0), PyArray_DIM(Pynp_ret, 1), CV_32F, PyArray_DATA(Pynp_ret));
            resize(pred, pred, cv::Size(ori_cols, ori_rows), 0, 0, CV_INTER_LINEAR);
            Py_DECREF(pParm);
            Py_DECREF(PyArray);
            Py_DECREF(pRetVal);
            Py_DECREF(Pynp_ret);

            depth = pred * mDepthRatio * focalLength / mTrainingFocalLength;

//            cout << "Finished depth prediction!" << endl << flush;
        }
    }

    DepthEstimator::~DepthEstimator() {
        Py_DECREF(mpModule);
        Py_DECREF(mpFunc);
        Py_Finalize();
    }
}
