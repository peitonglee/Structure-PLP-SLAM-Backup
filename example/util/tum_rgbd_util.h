/**
 * This file is part of Structure PLP-SLAM, originally from OpenVSLAM.
 *
 * Copyright 2022 DFKI (German Research Center for Artificial Intelligence)
 * Modified by Fangwen Shu <Fangwen.Shu@dfki.de>
 *
 * If you use this code, please cite the respective publications as
 * listed on the github repository.
 *
 * Structure PLP-SLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Structure PLP-SLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Structure PLP-SLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef EXAMPLE_UTIL_TUM_RGBD_UTIL_H
#define EXAMPLE_UTIL_TUM_RGBD_UTIL_H

#include <string>
#include <vector>

class tum_rgbd_sequence
{
public:
    struct frame
    {
        frame(const std::string &rgb_img_path, const std::string &depth_img_path, const double timestamp)
            : rgb_img_path_(rgb_img_path), depth_img_path_(depth_img_path), timestamp_(timestamp){};

        const std::string rgb_img_path_;
        const std::string depth_img_path_;
        const double timestamp_;
    };

    explicit tum_rgbd_sequence(const std::string &seq_dir_path, const double min_timediff_thr = 0.1);

    virtual ~tum_rgbd_sequence() = default;

    std::vector<frame> get_frames() const;

private:
    struct img_info
    {
        img_info(const double timestamp, const std::string &img_file_path)
            : timestamp_(timestamp), img_file_path_(img_file_path){};

        const double timestamp_;
        const std::string img_file_path_;
    };

    std::vector<img_info> acquire_image_information(const std::string &seq_dir_path,
                                                    const std::string &timestamp_file_path) const;

    std::vector<double> timestamps_;
    std::vector<std::string> rgb_img_file_paths_;
    std::vector<std::string> depth_img_file_paths_;
};

#endif // EXAMPLE_UTIL_TUM_RGBD_UTIL_H
