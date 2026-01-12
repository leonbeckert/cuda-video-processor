#pragma once

#include "common.h"
#include <string>
#include <memory>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

class VideoReader {
private:
    AVFormatContext* format_ctx;
    AVCodecContext* codec_ctx;
    const AVCodec* codec;
    AVFrame* frame;
    AVFrame* frame_rgb;
    AVPacket* packet;
    SwsContext* sws_ctx;
    int video_stream_index;
    uint8_t* buffer;

public:
    VideoReader();
    ~VideoReader();

    bool open(const std::string& filename);
    bool readFrame(VideoFrame& frame);
    void close();

    int getWidth() const;
    int getHeight() const;
    double getFPS() const;
    bool isOpen() const;
};
