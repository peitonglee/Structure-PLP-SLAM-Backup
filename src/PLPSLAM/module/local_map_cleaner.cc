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

#include "PLPSLAM/data/keyframe.h"
#include "PLPSLAM/data/landmark.h"
#include "PLPSLAM/data/landmark_line.h"
#include "PLPSLAM/module/local_map_cleaner.h"

#include <spdlog/spdlog.h>
namespace PLPSLAM
{
    namespace module
    {

        local_map_cleaner::local_map_cleaner(const bool is_monocular)
            : is_monocular_(is_monocular),
              _b_use_line_tracking(false)
        {
        }

        void local_map_cleaner::reset()
        {
            fresh_landmarks_.clear();

            if (_b_use_line_tracking)
            {
                _fresh_landmarks_line.clear();
            }
        }

        unsigned int local_map_cleaner::remove_redundant_landmarks(const unsigned int cur_keyfrm_id)
        {
            constexpr float observed_ratio_thr = 0.3;
            constexpr unsigned int num_reliable_keyfrms = 2;
            const unsigned int num_obs_thr = is_monocular_ ? 2 : 3;

            // states of observed landmarks
            enum class lm_state_t
            {
                Valid,
                Invalid,
                NotClear
            };

            unsigned int num_removed = 0;
            auto iter = fresh_landmarks_.begin();
            while (iter != fresh_landmarks_.end())
            {
                auto lm = *iter;

                // decide the state of lms the buffer
                auto lm_state = lm_state_t::NotClear;
                if (lm->will_be_erased())
                {
                    // in case `lm` will be erased
                    // remove `lm` from the buffer
                    lm_state = lm_state_t::Valid;
                }
                else if (lm->get_observed_ratio() < observed_ratio_thr)
                {
                    // if `lm` is not reliable
                    // remove `lm` from the buffer and the database
                    if (lm->get_Owning_Plane())
                    {
                        lm_state = lm_state_t::Valid;
                    }
                    else
                    {
                        lm_state = lm_state_t::Invalid;
                    }
                }
                else if (num_reliable_keyfrms + lm->first_keyfrm_id_ <= cur_keyfrm_id && lm->num_observations() <= num_obs_thr)
                {
                    // if the number of the observers of `lm` is small after some keyframes were inserted
                    // remove `lm` from the buffer and the database
                    if (lm->get_Owning_Plane())
                    {
                        lm_state = lm_state_t::Valid;
                    }
                    else
                    {
                        lm_state = lm_state_t::Invalid;
                    }
                }
                else if (num_reliable_keyfrms + 1U + lm->first_keyfrm_id_ <= cur_keyfrm_id)
                {
                    // if the number of the observers of `lm` is sufficient after some keyframes were inserted
                    // remove `lm` from the buffer
                    lm_state = lm_state_t::Valid;
                }

                // select to remove `lm` according to the state
                if (lm_state == lm_state_t::Valid)
                {
                    iter = fresh_landmarks_.erase(iter);
                }
                else if (lm_state == lm_state_t::Invalid)
                {
                    ++num_removed;

                    /*
                     * observations_.clear();
                     * will_be_erased_ = true;
                     */
                    lm->prepare_for_erasing();

                    iter = fresh_landmarks_.erase(iter);
                }
                else
                {
                    // hold decision because the state is NotClear
                    iter++;
                }
            }

            return num_removed;
        }

        unsigned int local_map_cleaner::remove_redundant_landmarks_line(const unsigned int cur_keyfrm_id)
        {
            constexpr float observed_ratio_thr = 0.3;
            constexpr unsigned int num_reliable_keyfrms = 2;
            const unsigned int num_obs_thr = 3;

            // states of observed landmarks
            enum class lm_state_t
            {
                Valid,
                Invalid,
                NotClear
            };

            unsigned int num_removed = 0;
            auto iter = _fresh_landmarks_line.begin();
            while (iter != _fresh_landmarks_line.end())
            {
                auto lm_line = *iter;

                // decide the state of lms the buffer
                auto lm_state = lm_state_t::NotClear;
                if (lm_line->will_be_erased())
                {
                    // 将被删除的线: 已经被移除了，不计入移除总数
                    // in case `lm` will be erased
                    // remove `lm` from the buffer
                    lm_state = lm_state_t::Valid;
                }
                else if (lm_line->get_observed_ratio() < observed_ratio_thr)
                {
                    // 线的观测率越大，表明该线越好，不应该被删除
                    // if `lm` is not reliable
                    // remove `lm` from the buffer and the database
                    lm_state = lm_state_t::Invalid;
                }
                else if (num_reliable_keyfrms + lm_line->_first_keyfrm_id <= cur_keyfrm_id && lm_line->num_observations() <= num_obs_thr)
                {
                    // 观测到3D线的帧必须大于等于3，且第一帧看到这根线的不能太早(与当前帧相差大于2帧)
                    // if the number of the observers of `lm` is small after some keyframes were inserted
                    // remove `lm` from the buffer and the database
                    lm_state = lm_state_t::Invalid;
                }
                else if (num_reliable_keyfrms + 1U + lm_line->_first_keyfrm_id <= cur_keyfrm_id)
                {
                    // 第一帧看到这根线的不能太早(与当前帧相差大于3帧)
                    // if the number of the observers of `lm` is sufficient after some keyframes were inserted
                    // remove `lm` from the buffer
                    lm_state = lm_state_t::Valid;
                }

                // select to remove `lm` according to the state
                if (lm_state == lm_state_t::Valid)
                {
                    iter = _fresh_landmarks_line.erase(iter);
                }
                else if (lm_state == lm_state_t::Invalid)
                {
                    ++num_removed;
                    lm_line->prepare_for_erasing();
                    iter = _fresh_landmarks_line.erase(iter);
                }
                else
                {
                    // hold decision because the state is NotClear
                    iter++;
                }
            }

            return num_removed;
        }

        unsigned int local_map_cleaner::remove_redundant_keyframes(data::keyframe *cur_keyfrm) const
        {
            // window size not to remove
            constexpr unsigned int window_size_not_to_remove = 2;
            // if the redundancy ratio of observations is larger than this threshold,
            // the corresponding keyframe will be erased
            constexpr float redundant_obs_ratio_thr = 0.9;

            unsigned int num_removed = 0;
            // check redundancy for each of the covisibilities
            const auto cur_covisibilities = cur_keyfrm->graph_node_->get_covisibilities();
            for (const auto covisibility : cur_covisibilities)
            {
                // cannot remove the origin
                // 不能删除初始化的关键帧
                if (covisibility->id_ == origin_keyfrm_id_)
                {
                    continue;
                }
                // cannot remove the recent keyframe(s)
                // 不能删除最近的关键帧
                if (covisibility->id_ <= cur_keyfrm->id_ && cur_keyfrm->id_ <= covisibility->id_ + window_size_not_to_remove)
                {
                    continue;
                }

                // count the number of redundant observations (num_redundant_obs) and valid observations (num_valid_obs)
                // for the covisibility
                unsigned int num_redundant_obs = 0;
                unsigned int num_valid_obs = 0;
                count_redundant_observations(covisibility, num_valid_obs, num_redundant_obs);

                // if the redundant observation ratio of `covisibility` is larger than the threshold, it will be removed
                // 如果该关键帧中的所有地标点中，过度追踪观测的关键点比例超过0.9，则认为该关键帧是冗余的
                if (redundant_obs_ratio_thr <= static_cast<float>(num_redundant_obs) / num_valid_obs)
                {
                    ++num_removed;
                    covisibility->prepare_for_erasing();
                }
            }

            return num_removed;
        }

        void local_map_cleaner::count_redundant_observations(data::keyframe *keyfrm, unsigned int &num_valid_obs, unsigned int &num_redundant_obs) const
        {
            // if the number of keyframes that observes the landmark with more reliable scale than the specified keyframe does,
            // it is considered as redundant
            constexpr unsigned int num_better_obs_thr = 3;

            num_valid_obs = 0;
            num_redundant_obs = 0;

            const auto landmarks = keyfrm->get_landmarks();
            // 遍历当前帧的所有地标点
            for (unsigned int idx = 0; idx < landmarks.size(); ++idx)
            {
                auto lm = landmarks.at(idx);
                if (!lm)
                {
                    continue;
                }
                if (lm->will_be_erased())
                {
                    continue;
                }

                // if depth is within the valid range, it won't be considered
                // 1. 如果当前地标点对应的双目测出的深度有效: 跳过
                // 2. 如果当前地标点对应的双目测出的深度无效，这个点是追踪到的点：
                //    - 认为该地标点是有效观测: ++num_valid_obs
                //    - 判断该地标点是否被三个或三个以上的关键帧看到过：
                //        - 没有三个关键帧看到过：认为该地标点不是冗余的
                //        - 被三个关键帧看到过：比较其余该地标点在其余关键帧中的观测尺度和在当前关键帧的观测尺度: cur_scale_level + 1 >= ngh_scale_level
                // 如果符合尺度规则：认为这个观测是好的观测，若在该地标点的所有观测中，好的观测大于等于3就认为该地标点是冗余的: ++num_redundant_obs

                const auto depth = keyfrm->depths_.at(idx);
                if (!is_monocular_ && (depth < 0.0 || keyfrm->depth_thr_ < depth))
                {
                    continue;
                }

                ++num_valid_obs;

                // if the number of the obs is smaller than the threshold, cannot remote the observers
                if (lm->num_observations() <= num_better_obs_thr)
                {
                    continue;
                }

                // `keyfrm` observes `lm` with the scale level `scale_level`
                const auto scale_level = keyfrm->undist_keypts_.at(idx).octave;
                // get observers of `lm`
                const auto observations = lm->get_observations();

                bool obs_by_keyfrm_is_redundant = false;

                // the number of the keyframes that observe `lm` with the more reliable (closer) scale
                unsigned int num_better_obs = 0;

                for (const auto obs : observations)
                {
                    const auto ngh_keyfrm = obs.first;
                    // 跳过自己对自己3D点的观测
                    if (*ngh_keyfrm == *keyfrm)
                    {
                        continue;
                    }

                    // `ngh_keyfrm` observes `lm` with the scale level `ngh_scale_level`
                    const auto ngh_scale_level = ngh_keyfrm->undist_keypts_.at(obs.second).octave;

                    // compare the scale levels
                    if (ngh_scale_level <= scale_level + 1)
                    {
                        // the observation by `ngh_keyfrm` is more reliable than `keyfrm`
                        ++num_better_obs;
                        if (num_better_obs_thr <= num_better_obs)
                        {
                            // if the number of the better observations is greater than the threshold,
                            // consider the observation of `lm` by `keyfrm` is redundant
                            obs_by_keyfrm_is_redundant = true;
                            break;
                        }
                    }
                }

                if (obs_by_keyfrm_is_redundant)
                {
                    ++num_redundant_obs;
                }
            }
        }

    } // namespace module
} // namespace PLPSLAM
