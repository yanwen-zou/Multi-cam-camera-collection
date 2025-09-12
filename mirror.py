from moviepy.editor import VideoFileClip, vfx

def mirror_video(input_path, output_path):
    """
    对视频进行左右镜像处理
    :param input_path: 输入视频路径
    :param output_path: 输出视频路径
    """
    # 读取视频 - 直接使用VideoFileClip
    clip = VideoFileClip(input_path)

    # 左右镜像（水平翻转） - 使用正确的语法
    flipped_clip = clip.fx(vfx.mirror_x)

    # 输出视频
    flipped_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=clip.fps
    )
    
    # 释放资源
    clip.close()
    flipped_clip.close()

if __name__ == "__main__":
    input_video = "0905_wild.mkv"        # 输入视频文件
    output_video = "mirror_ans.mp4"  # 输出视频文件
    mirror_video(input_video, output_video)
    print("左右镜像处理完成！")