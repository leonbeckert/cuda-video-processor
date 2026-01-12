#include "video_reader.h"
#include "common.h"
#include <iostream>
#include <cstring>

VideoReader::VideoReader()
    : format_ctx(nullptr), codec_ctx(nullptr), codec(nullptr),
      frame(nullptr), frame_rgb(nullptr), packet(nullptr),
      sws_ctx(nullptr), video_stream_index(-1), buffer(nullptr) {
}

VideoReader::~VideoReader() {
    close();
}

bool VideoReader::open(const std::string& filename) {
    if (avformat_open_input(&format_ctx, filename.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Could not open input file: " << filename << std::endl;
        return false;
    }

    if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
        std::cerr << "Could not find stream information" << std::endl;
        return false;
    }

    video_stream_index = -1;
    for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }

    if (video_stream_index == -1) {
        std::cerr << "Could not find video stream" << std::endl;
        return false;
    }

    AVCodecParameters* codec_params = format_ctx->streams[video_stream_index]->codecpar;

    codec = avcodec_find_decoder(codec_params->codec_id);
    if (!codec) {
        std::cerr << "Unsupported codec" << std::endl;
        return false;
    }

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        std::cerr << "Could not allocate codec context" << std::endl;
        return false;
    }

    if (avcodec_parameters_to_context(codec_ctx, codec_params) < 0) {
        std::cerr << "Could not copy codec parameters" << std::endl;
        return false;
    }

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        std::cerr << "Could not open codec" << std::endl;
        return false;
    }

    frame = av_frame_alloc();
    frame_rgb = av_frame_alloc();
    packet = av_packet_alloc();

    if (!frame || !frame_rgb || !packet) {
        std::cerr << "Could not allocate frames or packet" << std::endl;
        return false;
    }

    int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_RGBA, codec_ctx->width, codec_ctx->height, 1);
    buffer = (uint8_t*)av_malloc(num_bytes * sizeof(uint8_t));

    av_image_fill_arrays(frame_rgb->data, frame_rgb->linesize, buffer, AV_PIX_FMT_RGBA, codec_ctx->width, codec_ctx->height, 1);

    sws_ctx = sws_getContext(codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
                            codec_ctx->width, codec_ctx->height, AV_PIX_FMT_RGBA,
                            SWS_BILINEAR, nullptr, nullptr, nullptr);

    if (!sws_ctx) {
        std::cerr << "Could not initialize SWS context" << std::endl;
        return false;
    }

    const char* pix_fmt_name = av_get_pix_fmt_name(codec_ctx->pix_fmt);
    int bits_per_sample = av_get_bits_per_pixel(av_pix_fmt_desc_get(codec_ctx->pix_fmt));

    std::cout << "Successfully opened video: " << filename << std::endl;
    std::cout << "  Resolution: " << codec_ctx->width << "x" << codec_ctx->height << std::endl;
    std::cout << "  Pixel Format: " << (pix_fmt_name ? pix_fmt_name : "unknown") << std::endl;
    std::cout << "  Bits per pixel: " << bits_per_sample << std::endl;
    std::cout << "  Color space: " << av_color_space_name(codec_ctx->colorspace) << std::endl;
    std::cout << "  Color range: " << (codec_ctx->color_range == AVCOL_RANGE_JPEG ? "Full (0-255)" : "Limited (16-235)") << std::endl;

    return true;
}

bool VideoReader::readFrame(VideoFrame& frame) {
    while (av_read_frame(format_ctx, packet) >= 0) {
        if (packet->stream_index == video_stream_index) {
            int ret = avcodec_send_packet(codec_ctx, packet);
            if (ret < 0) {
                av_packet_unref(packet);
                continue;
            }

            ret = avcodec_receive_frame(codec_ctx, this->frame);
            if (ret == 0) {
                sws_scale(sws_ctx, this->frame->data, this->frame->linesize,
                         0, codec_ctx->height, frame_rgb->data, frame_rgb->linesize);

                frame = VideoFrame(codec_ctx->width, codec_ctx->height, 4);

                int bytes_per_line = codec_ctx->width * 4;
                for (int y = 0; y < codec_ctx->height; y++) {
                    memcpy(frame.data + y * bytes_per_line,
                           frame_rgb->data[0] + y * frame_rgb->linesize[0],
                           bytes_per_line);
                }

                av_packet_unref(packet);
                return true;
            }
        }
        av_packet_unref(packet);
    }

    return false;
}

void VideoReader::close() {
    if (sws_ctx) {
        sws_freeContext(sws_ctx);
        sws_ctx = nullptr;
    }

    if (buffer) {
        av_free(buffer);
        buffer = nullptr;
    }

    if (frame) {
        av_frame_free(&frame);
    }

    if (frame_rgb) {
        av_frame_free(&frame_rgb);
    }

    if (packet) {
        av_packet_free(&packet);
    }

    if (codec_ctx) {
        avcodec_free_context(&codec_ctx);
    }

    if (format_ctx) {
        avformat_close_input(&format_ctx);
    }

    video_stream_index = -1;
}

int VideoReader::getWidth() const {
    return codec_ctx ? codec_ctx->width : 0;
}

int VideoReader::getHeight() const {
    return codec_ctx ? codec_ctx->height : 0;
}

double VideoReader::getFPS() const {
    if (!format_ctx || video_stream_index < 0) {
        return 0.0;
    }

    AVRational fps = format_ctx->streams[video_stream_index]->r_frame_rate;
    return static_cast<double>(fps.num) / fps.den;
}

bool VideoReader::isOpen() const {
    return format_ctx != nullptr && codec_ctx != nullptr;
}
