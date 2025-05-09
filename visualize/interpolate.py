def interpolate(cls, pose1, pose2, alpha):
        """
        when alpha is 0, result is pose1
        when alpha is 1, result is pose2
        """
        skel = pose1.skel
        data = []
        for j in skel.joints:
            R1, p1 = conversions.T2Rp(pose1.get_transform(j, local=True))
            R2, p2 = conversions.T2Rp(pose2.get_transform(j, local=True))
            R, p = (
                math.slerp(R1, R2, alpha),
                math.lerp(p1, p2, alpha),
            )
            data.append(conversions.Rp2T(R, p))
        return Pose(pose1.skel, data)

    def get_pose_by_time(self, time):
        """
        If specified time is close to an integral multiple of (1/fps), returns
        the pose at that time. Else, returns an interpolated version
        """
        time = np.clip(time, 0, self.length())
        frame1 = self.time_to_frame(time)
        frame2 = min(frame1 + 1, self.num_frames() - 1)
        if frame1 == frame2:
            return self.poses[frame1]

        t1 = self.frame_to_time(frame1)
        t2 = self.frame_to_time(frame2)
        alpha = np.clip((time - t1) / (t2 - t1), 0.0, 1.0)

        return Pose.interpolate(self.poses[frame1], self.poses[frame2], alpha)