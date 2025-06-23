/*
 * Copyright (c) 2025 Vittorio Palmisano
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with FFmpeg; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "libavutil/avutil.h"
#include "libavutil/opt.h"
#include "libavutil/channel_layout.h"
#include "libavutil/samplefmt.h"
#include "libavfilter/avfilter.h"
#include "libavfilter/audio.h"
#include "libavutil/mem.h"
#include "libavutil/avstring.h"
#include "libavutil/internal.h"
#include "libavformat/avio.h"
#include "libavutil/thread.h"

#include "formats.h"

#include "whisper.h"

typedef struct WhisperContext
{
    const AVClass *class;
    char *model_path;
    char *language;
    bool use_gpu;
    int gpu_device;
    char *vad_model_path;
    float vad_threshold;
    int vad_min_speech_duration;
    int vad_min_silence_duration;

    int queue;
    char *destination;
    char *format;

    struct whisper_context *ctx_wsp;
    struct whisper_state *whisper_state;
    struct whisper_vad_context *ctx_vad;
    struct whisper_vad_params vad_params;

    float *audio_buffer;
    int audio_buffer_queue_size;
    int audio_buffer_fill_size;
    int audio_buffer_vad_size;

    int eof;
    int64_t next_pts;

    AVIOContext *avio_context;
    int index;
    int64_t timestamp;
} WhisperContext;

static void cb_log_disable(enum ggml_log_level, const char *, void *) {}

static int init(AVFilterContext *ctx)
{
    WhisperContext *wctx = ctx->priv;

    ggml_backend_load_all();
    whisper_log_set(cb_log_disable, NULL);

    // Init whisper context
    if (!wctx->model_path)
    {
        av_log(ctx, AV_LOG_ERROR, "No whisper model path specified. Use the 'model' option.\n");
        return AVERROR(EINVAL);
    }

    struct whisper_context_params params = whisper_context_default_params();
    params.use_gpu = wctx->use_gpu;
    params.gpu_device = wctx->gpu_device;

    wctx->ctx_wsp = whisper_init_from_file_with_params(wctx->model_path, params);
    if (wctx->ctx_wsp == NULL)
    {
        av_log(ctx, AV_LOG_ERROR, "Failed to initialize whisper context from model: %s\n", wctx->model_path);
        return AVERROR(EIO);
    }

    wctx->whisper_state = whisper_init_state(wctx->ctx_wsp);
    if (wctx->whisper_state == NULL)
    {
        av_log(ctx, AV_LOG_ERROR, "Failed to get whisper state from context\n");
        whisper_free(wctx->ctx_wsp);
        wctx->ctx_wsp = NULL;
        return AVERROR(EIO);
    }

    // Init VAD model context
    if (wctx->vad_model_path)
    {
        struct whisper_vad_context_params ctx_params = whisper_vad_default_context_params();
        ctx_params.n_threads = 4;
        // ctx_params.use_gpu = wctx->use_gpu; TODO (see: whisper_vad_init_context)
        ctx_params.gpu_device = wctx->gpu_device;
        wctx->ctx_vad = whisper_vad_init_from_file_with_params(
            wctx->vad_model_path,
            ctx_params);

        wctx->vad_params = whisper_vad_default_params();
        wctx->vad_params.threshold = wctx->vad_threshold;
        wctx->vad_params.min_speech_duration_ms = wctx->vad_min_speech_duration;
        wctx->vad_params.min_silence_duration_ms = wctx->vad_min_silence_duration;
        wctx->vad_params.max_speech_duration_s = (float)(wctx->audio_buffer_queue_size / 1000.0f);
        wctx->vad_params.speech_pad_ms = 0;
        wctx->vad_params.samples_overlap = 0;
    }

    // Init buffer
    wctx->audio_buffer_queue_size = WHISPER_SAMPLE_RATE * wctx->queue / 1000;
    wctx->audio_buffer = av_malloc(wctx->audio_buffer_queue_size * sizeof(float));
    if (!wctx->audio_buffer)
    {
        return AVERROR(ENOMEM);
    }

    wctx->audio_buffer_fill_size = 0;

    wctx->next_pts = AV_NOPTS_VALUE;

    wctx->avio_context = NULL;
    if (wctx->destination && strcmp("", wctx->destination))
    {
        int ret = 0;

        if (!strcmp("-", wctx->destination))
        {
            ret = avio_open(&wctx->avio_context, "pipe:1", AVIO_FLAG_WRITE);
        }
        else
        {
            ret = avio_open(&wctx->avio_context, wctx->destination, AVIO_FLAG_WRITE);
        }

        if (ret < 0)
        {
            av_log(ctx, AV_LOG_ERROR, "Could not open %s: %s\n",
                   wctx->destination, av_err2str(ret));
            return ret;
        }

        wctx->avio_context->direct = AVIO_FLAG_DIRECT;
    }

    av_log(ctx, AV_LOG_INFO, "Whisper filter initialized: model: %s lang: %s queue: %d ms\n",
           wctx->model_path, wctx->language, wctx->queue);

    return 0;
}

static void uninit(AVFilterContext *ctx)
{
    WhisperContext *wctx = ctx->priv;

    if (wctx->audio_buffer_fill_size > 0)
    {
        av_log(ctx, AV_LOG_WARNING, "Remaining audio buffer %d samples (%d seconds) after stopping\n",
               wctx->audio_buffer_fill_size,
               wctx->audio_buffer_fill_size / WHISPER_SAMPLE_RATE);
    }

    if (wctx->ctx_vad)
    {
        whisper_vad_free(wctx->ctx_vad);
        wctx->ctx_vad = NULL;
    }

    if (wctx->whisper_state)
    {
        whisper_free_state(wctx->whisper_state);
        wctx->whisper_state = NULL;
    }

    if (wctx->ctx_wsp)
    {
        whisper_free(wctx->ctx_wsp);
        wctx->ctx_wsp = NULL;
    }

    av_freep(&wctx->audio_buffer);

    if (wctx->avio_context)
    {
        avio_closep(&wctx->avio_context);
    }
}

static void run_transcription(AVFilterContext *ctx, AVDictionary **metadata, int end_pos)
{
    WhisperContext *wctx = ctx->priv;
    end_pos = FFMIN(end_pos, wctx->audio_buffer_fill_size);

    if (!wctx->ctx_wsp || end_pos == 0)
    {
        return;
    }

    if (!wctx->ctx_wsp)
    {
        return;
    }

    float duration = (float)end_pos / WHISPER_SAMPLE_RATE;

    av_log(ctx, AV_LOG_INFO, "run transcription %d/%d samples (%.2f seconds)...\n",
           end_pos, wctx->audio_buffer_fill_size,
           duration);

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.language = wctx->language;
    params.print_special = 0;
    params.print_progress = 0;
    params.print_realtime = 0;
    params.print_timestamps = 0;

    if (whisper_full(wctx->ctx_wsp, params, wctx->audio_buffer, end_pos) != 0)
    {
        av_log(ctx, AV_LOG_ERROR, "Failed to process audio with whisper.cpp\n");
        return;
    }

    const int n_segments = whisper_full_n_segments_from_state(wctx->whisper_state);
    char *segments_text = NULL;

    for (int i = 0; i < n_segments; ++i)
    {
        const bool turn = whisper_full_get_segment_speaker_turn_next(wctx->ctx_wsp, i);
        const int64_t t0 = whisper_full_get_segment_t0(wctx->ctx_wsp, i) * 10;
        const int64_t t1 = whisper_full_get_segment_t1(wctx->ctx_wsp, i) * 10;
        const char *text = whisper_full_get_segment_text(wctx->ctx_wsp, i);
        char *text_cleaned = av_strireplace(text + 1, "[BLANK_AUDIO]", "");

        if (av_strnlen(text_cleaned, 1) == 0)
        {
            av_free(text_cleaned);
            continue;
        }
        av_log(ctx, AV_LOG_INFO, "  [%ld-%ld%s]: \"%s\"\n", wctx->timestamp + t0, wctx->timestamp + t1, turn ? " (turn)" : "", text_cleaned);

        if (segments_text)
        {
            char *new_text = av_asprintf("%s%s", segments_text, text_cleaned);
            av_free(segments_text);
            segments_text = new_text;
        }
        else
        {
            segments_text = av_strdup(text_cleaned);
        }

        if (wctx->avio_context)
        {
            const int64_t start_t = wctx->timestamp + t0;
            const int64_t end_t = wctx->timestamp + t1;
            char *buf = NULL;

            if (!av_strcasecmp(wctx->format, "srt"))
            {
                buf = av_asprintf("%d\n%02ld:%02ld:%02ld.%03ld --> %02ld:%02ld:%02ld.%03ld\n%s\n\n",
                                  wctx->index,
                                  start_t / 3600000, (start_t / 60000) % 60, (start_t / 1000) % 60, start_t % 1000,
                                  end_t / 3600000, (end_t / 60000) % 60, (end_t / 1000) % 60, end_t % 1000,
                                  text_cleaned);
            }
            else if (!av_strcasecmp(wctx->format, "json"))
            {
                buf = av_asprintf("{\"start\":%ld,\"end\":%ld,\"text\":\"%s\",\"turn\":%s}\n",
                                  start_t, end_t, text_cleaned, turn ? "true" : "false");
            }
            else
            {
                buf = av_strdup(text_cleaned);
            }

            if (buf)
            {
                avio_write(wctx->avio_context, buf, strlen(buf));
                av_free(buf);
            }
        }

        av_free(text_cleaned);
    }

    wctx->index++;
    wctx->timestamp += (int64_t)(duration * 1000);

    if (metadata && segments_text)
    {
        av_dict_set(metadata, "lavfi.whisper.text", segments_text, 0);
        char *duration_text = av_asprintf("%f", duration);
        av_dict_set(metadata, "lavfi.whisper.duration", duration_text, 0);
        av_free(duration_text);
    }
    if (segments_text)
    {
        av_free(segments_text);
    }

    memcpy(wctx->audio_buffer, wctx->audio_buffer + end_pos, end_pos * sizeof(float));
    wctx->audio_buffer_fill_size -= end_pos;
    wctx->audio_buffer_vad_size = wctx->audio_buffer_fill_size;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    WhisperContext *wctx = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    AVDictionary **metadata = &frame->metadata;

    const int samples = frame->nb_samples;
    const float *input_data = (const float *)frame->data[0];

    if (wctx->audio_buffer_fill_size + samples > wctx->audio_buffer_queue_size)
    {
        run_transcription(ctx, metadata, wctx->audio_buffer_fill_size);
    }

    memcpy(wctx->audio_buffer + wctx->audio_buffer_fill_size, input_data, samples * sizeof(float));
    wctx->audio_buffer_fill_size += samples;

    if (wctx->ctx_vad && (wctx->audio_buffer_fill_size - wctx->audio_buffer_vad_size) >=
                             WHISPER_SAMPLE_RATE * (wctx->vad_min_speech_duration + wctx->vad_min_silence_duration) / 1000)
    {
        struct whisper_vad_segments *segments = whisper_vad_segments_from_samples(
            wctx->ctx_vad, wctx->vad_params, wctx->audio_buffer, wctx->audio_buffer_fill_size);
        wctx->audio_buffer_vad_size = wctx->audio_buffer_fill_size;

        if (!segments)
        {
            av_log(ctx, AV_LOG_ERROR, "failed to detect VAD\n");
        }
        else
        {
            int n_segments = whisper_vad_segments_n_segments(segments);

            if (n_segments > 0)
            {
                const int64_t start_ms = whisper_vad_segments_get_segment_t0(segments, n_segments - 1) * 10;
                const int64_t end_ms = whisper_vad_segments_get_segment_t1(segments, n_segments - 1) * 10;
                int end_pos = (int)(end_ms * WHISPER_SAMPLE_RATE / 1000);

                if (end_pos < wctx->audio_buffer_fill_size)
                {
                    av_log(ctx, AV_LOG_INFO, "VAD detected %d segments, start: %ld ms, end: %ld ms (buffer: %d ms)\n",
                           n_segments, start_ms, end_ms, 1000 * wctx->audio_buffer_fill_size / WHISPER_SAMPLE_RATE);
                    run_transcription(ctx, metadata, end_pos);
                }
            }

            whisper_vad_free_segments(segments);
        }
    }
    else if (wctx->audio_buffer_fill_size >= wctx->audio_buffer_queue_size)
    {
        run_transcription(ctx, metadata, wctx->audio_buffer_fill_size);
    }

    wctx->next_pts = frame->pts + av_rescale_q(frame->nb_samples, (AVRational){1, inlink->sample_rate}, inlink->time_base);
    return ff_filter_frame(outlink, frame);
}

static int push_last_frame(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    WhisperContext *wctx = ctx->priv;
    AVFrame *frame;
    int n_out = 1;

    if (ctx->is_disabled || wctx->audio_buffer_fill_size == 0)
        return 0;
    frame = ff_get_audio_buffer(outlink, n_out);
    if (!frame)
        return AVERROR(ENOMEM);

    av_samples_set_silence(frame->extended_data, 0,
                           n_out,
                           frame->ch_layout.nb_channels,
                           frame->format);

    frame->pts = wctx->next_pts;
    if (wctx->next_pts != AV_NOPTS_VALUE)
        wctx->next_pts += av_rescale_q(n_out, (AVRational){1, outlink->sample_rate}, outlink->time_base);

    run_transcription(ctx, &frame->metadata, wctx->audio_buffer_fill_size);

    return ff_filter_frame(outlink, frame);
}

static int activate(AVFilterContext *ctx)
{
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    WhisperContext *wctx = ctx->priv;
    int64_t pts;
    int status;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    if (!wctx->eof && ff_inlink_queued_frames(inlink))
    {
        AVFrame *frame = NULL;
        int ret;

        ret = ff_inlink_consume_frame(inlink, &frame);
        if (ret < 0)
            return ret;
        if (ret > 0)
            return filter_frame(inlink, frame);
    }

    if (!wctx->eof && ff_inlink_acknowledge_status(inlink, &status, &pts))
        wctx->eof = status == AVERROR_EOF;

    if (wctx->eof)
    {
        push_last_frame(outlink);

        ff_outlink_set_status(outlink, AVERROR_EOF, wctx->next_pts);
        return 0;
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink);

    return FFERROR_NOT_READY;
}

#define OFFSET(x) offsetof(WhisperContext, x)
#define FLAGS AV_OPT_FLAG_AUDIO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption whisper_options[] = {
    {"model", "Path to the whisper.cpp model file", OFFSET(model_path), AV_OPT_TYPE_STRING, .flags = FLAGS},
    {"language", "Language for transcription ('auto' for auto-detect)", OFFSET(language), AV_OPT_TYPE_STRING, {.str = "auto"}, .flags = FLAGS},
    {"queue", "Audio queue size in milliseconds", OFFSET(queue), AV_OPT_TYPE_INT, {.i64 = 3000}, 20, INT_MAX, .flags = FLAGS},
    {"use_gpu", "Use GPU for processing", OFFSET(use_gpu), AV_OPT_TYPE_BOOL, {.i64 = 1}, 0, 1, .flags = FLAGS},
    {"gpu_device", "GPU device to use", OFFSET(gpu_device), AV_OPT_TYPE_INT, {.i64 = 0}, 0, INT_MAX, .flags = FLAGS},
    {"destination", "Output destination", OFFSET(destination), AV_OPT_TYPE_STRING, {.str = ""}, .flags = FLAGS},
    {"format", "Output format (text|srt|json)", OFFSET(format), AV_OPT_TYPE_STRING, {.str = "text"}, .flags = FLAGS},
    {"vad_model", "Path to the VAD model file", OFFSET(vad_model_path), AV_OPT_TYPE_STRING, .flags = FLAGS},
    {"vad_threshold", "VAD threshold", OFFSET(vad_threshold), AV_OPT_TYPE_FLOAT, {.dbl = 0.5}, 0.0, 1.0, .flags = FLAGS},
    {"vad_min_speech_duration", "Minimum speech duration in milliseconds for VAD", OFFSET(vad_min_speech_duration), AV_OPT_TYPE_INT, {.i64 = 50}, 20, INT_MAX, .flags = FLAGS},
    {"vad_min_silence_duration", "Minimum silence duration in milliseconds for VAD", OFFSET(vad_min_silence_duration), AV_OPT_TYPE_INT, {.i64 = 500}, 0, INT_MAX, .flags = FLAGS},
    {NULL}};

static const AVClass whisper_class = {
    .class_name = "whisper",
    .item_name = av_default_item_name,
    .option = whisper_options,
    .version = LIBAVUTIL_VERSION_INT,
};

static const AVFilterPad whisper_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_AUDIO,
    },
};

const FFFilter ff_af_whisper = {
    .p.name = "whisper",
    .p.description = NULL_IF_CONFIG_SMALL("Transcribe audio using whisper.cpp."),
    .p.priv_class = &whisper_class,
    .p.flags = AVFILTER_FLAG_METADATA_ONLY,
    .init = init,
    .uninit = uninit,
    .activate = activate,
    .priv_size = sizeof(WhisperContext),
    FILTER_INPUTS(ff_audio_default_filterpad),
    FILTER_OUTPUTS(whisper_outputs),
    FILTER_SAMPLEFMTS(AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_FLTP),
};
