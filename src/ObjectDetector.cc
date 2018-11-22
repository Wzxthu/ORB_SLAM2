
#include <include/ObjectDetector.h>

#include "ObjectDetector.h"
#include "darknet.h"

#define NETWORK(p) ((network *)p)

namespace ORB_SLAM2 {

ObjectDetector::ObjectDetector(
        const char *cfgfile,
        const char *weightfile,
        float nms,
        float thresh,
        float hierThresh) : mNms(nms), mThresh(thresh), mHierThresh(hierThresh) {
    mpNet = load_network(cfgfile, weightfile, 0);
    set_batch_network(NETWORK(mpNet), 1);
}

void ObjectDetector::Detect(const cv::Mat &im, std::vector<Object> &objects) {
    clock_t time;
    network *net = NETWORK(mpNet);

    layer l = net->layers[net->n - 1];

    image im_dn = mat_to_image(im);
    image sized = letterbox_image(im_dn, net->w, net->h);
    float *X = sized.data;
    time = clock();
    network_predict(net, X);

    int nboxes = 0;
    detection *dets = get_network_boxes(net, im_dn.w, im_dn.h, mThresh, mHierThresh, 0, 1, &nboxes);
    if (mNms) do_nms_sort(dets, nboxes, l.classes, mNms);
    printf("Predicted %d boxes in %f seconds.\n", nboxes, sec(clock() - time));

    objects.clear();
    objects.reserve(static_cast<unsigned long>(nboxes));
    for (int i = 0; i < nboxes; ++i) {
        int most_likely_class = -1;
        float most_likely_prob = 0;
        for (int k = 0; k < dets[i].classes; ++k) {
            if (dets[i].prob[k] > most_likely_prob) {
                most_likely_prob = dets[i].prob[k];
                most_likely_class = k;
            }
        }
        objects.emplace_back((Object) {
                .bbox = cv::Rect(static_cast<int>(dets[i].bbox.x * im.cols),
                                 static_cast<int>(dets[i].bbox.y * im.rows),
                                 static_cast<int>(dets[i].bbox.w * im.cols),
                                 static_cast<int>(dets[i].bbox.h * im.rows)),
                .classes = dets[i].classes,
                .objectness = dets[i].objectness,
                .classIdx = most_likely_class,
        });
    }

//    char **names = get_labels("Thirdparty/darknet/data/coco.names");
//    draw_detections(im_dn, dets, nboxes, mThresh, names, alphabet, l.classes);
//    show_image(im_dn, "predictions", 0);

    free_detections(dets, nboxes);
    free_image(im_dn);
    free_image(sized);
}

}