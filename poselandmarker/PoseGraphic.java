
package com.example.Base;

import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.StrictMath.atan2;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;

import com.ai.pose.PoseClassList;
import com.example.cervixcenter.Ai.GraphicOverlay;
import com.example.cervixcenter.Utils.Constants;
import com.example.cervixcenter.Utils.SharedPreferencesUtils;
import com.google.common.primitives.Ints;
import com.google.mlkit.vision.common.PointF3D;
import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseLandmark;

import org.greenrobot.eventbus.EventBus;

import java.util.Date;
import java.util.List;
import java.util.Locale;

/** Draw the detected pose in preview. */
public class PoseGraphic extends GraphicOverlay.Graphic {
    private static final float DOT_RADIUS = 8.0f;

    private static final float IN_FRAME_LIKELIHOOD_TEXT_SIZE = 0.0f;
    private static final float STROKE_WIDTH = 10.0f;

    private static final float POSE_CLASSIFICATION_TEXT_SIZE = 60.0f;

    private final Pose pose;
    private final boolean showInFrameLikelihood;
    private final boolean visualizeZ;
    private final boolean rescaleZForVisualization;
    private float zMin = Float.MAX_VALUE;
    private float zMax = Float.MIN_VALUE;

    private final List<String> poseClassification;
    private final Paint classificationTextPaint;
    private final Paint leftPaint;
    private final Paint rightPaint;
    private final Paint whitePaint;

    private double  YWangle,TTangel;
    private double  Qtangle;
    Date endDate,YWStartDate,StartDate;


    PoseGraphic(
            GraphicOverlay overlay,
            Pose pose,
            boolean showInFrameLikelihood,
            boolean visualizeZ,
            boolean rescaleZForVisualization,
            List<String> poseClassification) {
        super(overlay);
        this.pose = pose;
        this.showInFrameLikelihood = showInFrameLikelihood;
        this.visualizeZ = visualizeZ;
        this.rescaleZForVisualization = rescaleZForVisualization;

        this.poseClassification = poseClassification;
        classificationTextPaint = new Paint();
        classificationTextPaint.setColor(Color.WHITE);
        classificationTextPaint.setTextSize(POSE_CLASSIFICATION_TEXT_SIZE);
        classificationTextPaint.setShadowLayer(5.0f, 0f, 0f, Color.BLACK);

        whitePaint = new Paint();
        whitePaint.setStrokeWidth(STROKE_WIDTH);
        whitePaint.setColor(Color.WHITE);
        whitePaint.setTextSize(IN_FRAME_LIKELIHOOD_TEXT_SIZE);
        leftPaint = new Paint();
        leftPaint.setStrokeWidth(STROKE_WIDTH);
        leftPaint.setColor(Color.GREEN);
        rightPaint = new Paint();
        rightPaint.setStrokeWidth(STROKE_WIDTH);
        rightPaint.setColor(Color.YELLOW);
    }

    @Override
    public void draw(Canvas canvas) {
        List<PoseLandmark> landmarks = pose.getAllPoseLandmarks();
        if (landmarks.isEmpty()) {
            return;
        }

        float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
        for (int i = 0; i < poseClassification.size(); i++) {
            float classificationY = (canvas.getHeight() - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f
                    * (poseClassification.size() - i));
            canvas.drawText(
                    poseClassification.get(i),
                    classificationX,
                    classificationY,
                    classificationTextPaint);
        }

        for (PoseLandmark landmark : landmarks) {
            drawPoint(canvas, landmark, whitePaint);
            if (visualizeZ && rescaleZForVisualization) {
                zMin = min(zMin, landmark.getPosition3D().getZ());
                zMax = max(zMax, landmark.getPosition3D().getZ());
            }
        }

        PoseLandmark nose = pose.getPoseLandmark(PoseLandmark.NOSE);
        PoseLandmark lefyEyeInner = pose.getPoseLandmark(PoseLandmark.LEFT_EYE_INNER);
        PoseLandmark lefyEye = pose.getPoseLandmark(PoseLandmark.LEFT_EYE);
        PoseLandmark leftEyeOuter = pose.getPoseLandmark(PoseLandmark.LEFT_EYE_OUTER);
        PoseLandmark rightEyeInner = pose.getPoseLandmark(PoseLandmark.RIGHT_EYE_INNER);
        PoseLandmark rightEye = pose.getPoseLandmark(PoseLandmark.RIGHT_EYE);
        PoseLandmark rightEyeOuter = pose.getPoseLandmark(PoseLandmark.RIGHT_EYE_OUTER);
        PoseLandmark leftEar = pose.getPoseLandmark(PoseLandmark.LEFT_EAR);
        PoseLandmark rightEar = pose.getPoseLandmark(PoseLandmark.RIGHT_EAR);
        PoseLandmark leftMouth = pose.getPoseLandmark(PoseLandmark.LEFT_MOUTH);
        PoseLandmark rightMouth = pose.getPoseLandmark(PoseLandmark.RIGHT_MOUTH);

        PoseLandmark lefthandlder = pose.getPoseLandmark(PoseLandmark.LEFT_handLDER);
        PoseLandmark righthandlder = pose.getPoseLandmark(PoseLandmark.RIGHT_handLDER);
        PoseLandmark leftElbow = pose.getPoseLandmark(PoseLandmark.LEFT_ELBOW);
        PoseLandmark rightElbow = pose.getPoseLandmark(PoseLandmark.RIGHT_ELBOW);
        PoseLandmark leftWrist = pose.getPoseLandmark(PoseLandmark.LEFT_WRIST);
        PoseLandmark rightWrist = pose.getPoseLandmark(PoseLandmark.RIGHT_WRIST);
        PoseLandmark leftHip = pose.getPoseLandmark(PoseLandmark.LEFT_HIP);
        PoseLandmark rightHip = pose.getPoseLandmark(PoseLandmark.RIGHT_HIP);
        PoseLandmark leftKnee = pose.getPoseLandmark(PoseLandmark.LEFT_KNEE);
        PoseLandmark rightKnee = pose.getPoseLandmark(PoseLandmark.RIGHT_KNEE);
        PoseLandmark leftAnkle = pose.getPoseLandmark(PoseLandmark.LEFT_ANKLE);
        PoseLandmark rightAnkle = pose.getPoseLandmark(PoseLandmark.RIGHT_ANKLE);

        PoseLandmark leftPinky = pose.getPoseLandmark(PoseLandmark.LEFT_PINKY);
        PoseLandmark rightPinky = pose.getPoseLandmark(PoseLandmark.RIGHT_PINKY);
        PoseLandmark leftIndex = pose.getPoseLandmark(PoseLandmark.LEFT_INDEX);
        PoseLandmark rightIndex = pose.getPoseLandmark(PoseLandmark.RIGHT_INDEX);
        PoseLandmark leftThumb = pose.getPoseLandmark(PoseLandmark.LEFT_THUMB);
        PoseLandmark rightThumb = pose.getPoseLandmark(PoseLandmark.RIGHT_THUMB);
        PoseLandmark leftHeel = pose.getPoseLandmark(PoseLandmark.LEFT_HEEL);
        PoseLandmark rightHeel = pose.getPoseLandmark(PoseLandmark.RIGHT_HEEL);
        PoseLandmark leftFootIndex = pose.getPoseLandmark(PoseLandmark.LEFT_FOOT_INDEX);
        PoseLandmark rightFootIndex = pose.getPoseLandmark(PoseLandmark.RIGHT_FOOT_INDEX);


        switch (PName) {
            case "S-P":

                if((boolean)SharedPreferencesUtils.getParam(MyApp.sContext, "qsjs", false)){
                    if (pose.getAllPoseLandmarks().get(11).getPosition3D().getZ() > pose.getAllPoseLandmarks().get(12).getPosition3D().getZ()) {
                        if (pose.getAllPoseLandmarks().get(26).getPosition3D().getY()-pose.getAllPoseLandmarks().get(24).getPosition3D().getY()>=35) {
                            Log.d("TAG","end");
                        }
                    }else{
                        if (pose.getAllPoseLandmarks().get(25).getPosition3D().getY()-pose.getAllPoseLandmarks().get(23).getPosition3D().getY()>=35) {
                            Log.d("TAG","end");
                        }
                    }

                }



                switch (dy) {

                    case "N1":

                        if (PoseClassList.ZX(pose.getAllPoseLandmarks().get(28).getPosition3D().getY(),pose.getAllPoseLandmarks().get(24).getPosition3D().getY())) {
                            MyApp.Down = true;
                        } else {
                            MyApp.Down = false;
                        }

                        if (pose.getAllPoseLandmarks().get(11).getPosition3D().getZ() > pose.getAllPoseLandmarks().get(12).getPosition3D().getZ()) {
                            YWangle = getAngle(pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(24), pose.getAllPoseLandmarks().get(28));
                            MyApp.XGY=pose.getAllPoseLandmarks().get(26).getPosition3D().getY();
                        } else {
                            YWangle = getAngle(pose.getAllPoseLandmarks().get(11), pose.getAllPoseLandmarks().get(23), pose.getAllPoseLandmarks().get(27));
                        }
                        if (MyApp.Down && YWangle > 160 && YWangle < 200) {
                            MyApp.TPtype = true;
                        }
                        if (PoseClassList.JS(MyApp.Down,YWangle,MyApp.TPtype)) {
                            StartDate = new Date(System.currentTimeMillis());
                            if (StartDate.getTime() - MyApp.EndDate.getTime() > 400) {
                                Log.d("TAG","good");
                            }
                        }
                        break;
                    case "N2":

                        if (PoseClassList.ZX(pose.getAllPoseLandmarks().get(28).getPosition3D().getY(),pose.getAllPoseLandmarks().get(24).getPosition3D().getY())) {
                            MyApp.Down = true;
                        } else {
                            MyApp.Down = false;
                        }

                        MyApp.YXjiaodu=getAngle(pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(24));


                        if (MyApp.Down && pose.getAllPoseLandmarks().get(11).getPosition3D().getZ() > pose.getAllPoseLandmarks().get(12).getPosition3D().getZ()) {
                            YWangle = getAngle(pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(24), pose.getAllPoseLandmarks().get(28));

                        } else {
                            YWangle = getAngle(pose.getAllPoseLandmarks().get(11), pose.getAllPoseLandmarks().get(23), pose.getAllPoseLandmarks().get(27));
                        }
                        if (MyApp.Down && YWangle > 140 && YWangle < 200) {
                            MyApp.YWtype = true;
                        }
                        if (MyApp.Down && MyApp.TPtype && YWangle <= 130) {
                            MyApp.QZtype = true;
                        }

                        if (YWangle > 170 && YWangle < 200) {
                            if(PoseClassList.DX(pose.getAllPoseLandmarks().get(11).getPosition3D().getZ(),pose.getAllPoseLandmarks().get(12).getPosition3D().getZ())){
                                if (PoseClassList.DX1(pose.getAllPoseLandmarks().get(12).getPosition3D().getY(),pose.getAllPoseLandmarks().get(30).getPosition3D().getY())|| PoseClassList.DX2(pose.getAllPoseLandmarks().get(12).getPosition3D().getY(),pose.getAllPoseLandmarks().get(30).getPosition3D().getY())) {
                                    MyApp.TPtype = true;
                                    MyApp.YWtype = true;
                                }
                            }else{
                                if (PoseClassList.DX1(pose.getAllPoseLandmarks().get(11).getPosition3D().getY(),pose.getAllPoseLandmarks().get(29).getPosition3D().getY()) || PoseClassList.DX2(pose.getAllPoseLandmarks().get(11).getPosition3D().getY(),pose.getAllPoseLandmarks().get(29).getPosition3D().getY())) {
                                    MyApp.TPtype = true;
                                    MyApp.YWtype = true;
                                }
                            }
                        }

                        if (PoseClassList.SS(YWangle,YWangle,pose.getAllPoseLandmarks().get(11).getPosition3D().getZ(),pose.getAllPoseLandmarks().get(12).getPosition3D().getZ())) {


                            if (YWangle < 150) {

                                if (PoseClassList.SS1(getAngle(pose.getAllPoseLandmarks().get(16), pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(12)))) {
                                    if (PoseClassList.SS2(pose.getAllPoseLandmarks().get(16).getPosition3D().getX(), pose.getAllPoseLandmarks().get(14).getPosition3D().getX(), MyApp.TPtype)) {
                                        endDate = new Date(System.currentTimeMillis());
                                        if (PoseClassList.SS3(endDate.getTime(), MyApp.curDate.getTime())) {
                                            MyApp.curDate = endDate;
                                            if (MyApp.Yheadnumber <= 0) {
                                                Yhead = true;
                                            } else {
                                                MyApp.Yheadnumber--;
                                            }
                                        }
                                    }

                                }
                            }
                        } else if(YWangle<150&&YWangle>100&&pose.getAllPoseLandmarks().get(11).getPosition3D().getZ() < pose.getAllPoseLandmarks().get(12).getPosition3D().getZ()) {

                            double SSYoujiaodu = getAngle(pose.getAllPoseLandmarks().get(11), pose.getAllPoseLandmarks().get(13), pose.getAllPoseLandmarks().get(15));

                            if(YWangle < 150){

                                if(PoseClassList.SS1(getAngle(pose.getAllPoseLandmarks().get(15), pose.getAllPoseLandmarks().get(13), pose.getAllPoseLandmarks().get(11)))){

                                    if (PoseClassList.SS2(pose.getAllPoseLandmarks().get(15).getPosition3D().getX(), pose.getAllPoseLandmarks().get(13).getPosition3D().getX(), MyApp.TPtype)) {
                                        endDate = new Date(System.currentTimeMillis());
                                        if (PoseClassList.SS3(endDate.getTime(), MyApp.curDate.getTime())) {
                                            MyApp.curDate = endDate;
                                            if (MyApp.Yheadnumber <= 0) {
                                                Yhead = true;
                                            } else {
                                                MyApp.Yheadnumber--;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if (PoseClassList.SS4(YWangle,MyApp.TPtype,MyApp.Down)) {
                            MyApp.SP = true;
                            if (PoseClassList.ZX1(MyApp.Down,pose.getAllPoseLandmarks().get(11).getPosition3D().getZ(),pose.getAllPoseLandmarks().get(12).getPosition3D().getZ())){
                                if(PoseClassList.DX2(pose.getAllPoseLandmarks().get(20).getPosition3D().getY(),pose.getAllPoseLandmarks().get(10).getPosition3D().getY())){
                                    Yhead=true;
                                }
                            } else {
                                if(PoseClassList.DX2(pose.getAllPoseLandmarks().get(19).getPosition3D().getY(),pose.getAllPoseLandmarks().get(9).getPosition3D().getY())){
                                    Yhead=true;
                                }
                            }

                            if (hips) {
                                Log.d("TAG","hipsnumber");
                            } else if (true) {
                                if (Yhead||MyApp.Swing) {
                                    Log.d("TAG","Yhead");
                                } else {
                                    Log.d("TAG","good");
                                }
                            }
                        } else if (PoseClassList.SS5(YWangle,MyApp.TPtype,MyApp.YWtype,MyApp.Down)) {
                            YWStartDate = new Date(System.currentTimeMillis());
                            if (YWStartDate.getTime() - MyApp.endYW.getTime() > 400) {
                                Log.d("TAG","bad");
                            }
                        } else if (PoseClassList.SS6(MyApp.TPtype,MyApp.QZtype,MyApp.Down,YWangle,YWangle)) {

                            YWStartDate = new Date(System.currentTimeMillis());
                            if (YWStartDate.getTime() - MyApp.endYW.getTime() > 400) {
                                Log.d("TAG","bad");
                            }
                        }

                        break;


                    case "N3":

                        if (PoseClassList.DX(pose.getAllPoseLandmarks().get(11).getPosition3D().getZ(),pose.getAllPoseLandmarks().get(12).getPosition3D().getZ())) {
                            MyApp.YXjiaodu=getAngle(pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(24));
                        }else{
                            MyApp.YXjiaodu=getAngle(pose.getAllPoseLandmarks().get(13), pose.getAllPoseLandmarks().get(11), pose.getAllPoseLandmarks().get(23));
                        }


                        if (pose.getAllPoseLandmarks().get(30).getInFrameLikelihood() > 0.98 && pose.getAllPoseLandmarks().get(24).getInFrameLikelihood() > 0.98) {
                            if (PoseClassList.ZX(pose.getAllPoseLandmarks().get(30).getPosition3D().getY(),pose.getAllPoseLandmarks().get(24).getPosition3D().getY())) {
                                MyApp.Down = true;
                            } else {
                                MyApp.Down = false;
                            }
                        }

                        if (PoseClassList.ZX2(MyApp.Down,pose.getAllPoseLandmarks().get(30).getPosition3D().getY(),pose.getAllPoseLandmarks().get(24).getPosition3D().getY())) {
                        } else {
                            MyApp.Down = false;
                        }


                        if (PoseClassList.ZX1(MyApp.Down,pose.getAllPoseLandmarks().get(11).getPosition3D().getZ(),pose.getAllPoseLandmarks().get(12).getPosition3D().getZ())) {
                            YWangle = getAngle(pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(24), pose.getAllPoseLandmarks().get(28));
                            Qtangle = getAngle(pose.getAllPoseLandmarks().get(24), pose.getAllPoseLandmarks().get(26), pose.getAllPoseLandmarks().get(28));
                        } else {
                            YWangle = getAngle(pose.getAllPoseLandmarks().get(11), pose.getAllPoseLandmarks().get(23), pose.getAllPoseLandmarks().get(27));
                            Qtangle = getAngle(pose.getAllPoseLandmarks().get(23), pose.getAllPoseLandmarks().get(25), pose.getAllPoseLandmarks().get(27));
                        }

                        if (PoseClassList.DX3(MyApp.Down,YWangle,YWangle)) {
                            MyApp.YWtype = true;
                        }

                        if (PoseClassList.DX4(MyApp.Down,MyApp.TPtype,YWangle)) {
                            MyApp.QZtype = true;
                        }


                        if (PoseClassList.DX5(YWangle,YWangle)) {

                            if (PoseClassList.DX1(pose.getAllPoseLandmarks().get(12).getPosition3D().getY(),pose.getAllPoseLandmarks().get(30).getPosition3D().getY())|| PoseClassList.DX2(pose.getAllPoseLandmarks().get(12).getPosition3D().getY(),pose.getAllPoseLandmarks().get(30).getPosition3D().getY())) {
                                MyApp.TPtype = true;
                                MyApp.YWtype = true;
                                if(PoseClassList.DX6(Qtangle)){
                                    MyApp.bend = false;
                                }else{
                                    MyApp.bend = true;
                                }

                            }
                        }

                        if (pose.getAllPoseLandmarks().get(30).getInFrameLikelihood() > 0.98 && pose.getAllPoseLandmarks().get(24).getInFrameLikelihood() > 0.98) {
                            if (PoseClassList.DX(pose.getAllPoseLandmarks().get(11).getPosition3D().getZ(),pose.getAllPoseLandmarks().get(12).getPosition3D().getZ())) {
                                TTangel = getAngle(pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(24), pose.getAllPoseLandmarks().get(26));
                            } else {
                                TTangel = getAngle(pose.getAllPoseLandmarks().get(11), pose.getAllPoseLandmarks().get(23), pose.getAllPoseLandmarks().get(25));
                            }
                            if (PoseClassList.DX7(MyApp.TPtype,TTangel)) {
                                hips = true;
                            } else {

                            }
                        }


                        if (YWangle<155&&YWangle>100&&pose.getAllPoseLandmarks().get(11).getPosition3D().getZ() > pose.getAllPoseLandmarks().get(12).getPosition3D().getZ()) {


                            double SSYoujiaodu = getAngle(pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(16));


                            if(YWangle < 150) {
                                if(YWangle < 120){
                                    if(PoseClassList.SS7(pose.getAllPoseLandmarks().get(16).getPosition3D().getY(),pose.getAllPoseLandmarks().get(10).getPosition3D().getY())){
                                        Yhead = true;
                                    }
                                }

                                if (PoseClassList.SS2(pose.getAllPoseLandmarks().get(16).getPosition3D().getX(),pose.getAllPoseLandmarks().get(14).getPosition3D().getX(),MyApp.TPtype)) {

                                    endDate = new Date(System.currentTimeMillis());
                                    if (PoseClassList.SS8(endDate.getTime(),MyApp.curDate.getTime())) {
                                        MyApp.curDate = endDate;
                                        if (MyApp.Yheadnumber == 0) {
                                            Yhead = true;
                                        } else {
                                            MyApp.Yheadnumber--;
                                        }
                                    }
                                }
                            }

                        } else if(PoseClassList.SS9(YWangle,YWangle,pose.getAllPoseLandmarks().get(11).getPosition3D().getZ(),pose.getAllPoseLandmarks().get(12).getPosition3D().getZ())) {


                            if(YWangle < 160){
                                if(YWangle < 120){
                                    if(PoseClassList.SS7(pose.getAllPoseLandmarks().get(15).getPosition3D().getY(),pose.getAllPoseLandmarks().get(9).getPosition3D().getY())){
                                        Yhead = true;

                                    }
                                }

                                if (PoseClassList.SS2(pose.getAllPoseLandmarks().get(15).getPosition3D().getX(),pose.getAllPoseLandmarks().get(13).getPosition3D().getX(),MyApp.TPtype)) {

                                    endDate = new Date(System.currentTimeMillis());
                                    if (PoseClassList.SS8(endDate.getTime(), MyApp.curDate.getTime())) {
                                        MyApp.curDate = endDate;
                                        if (MyApp.Yheadnumber == 0) {
                                            Yhead = true;
                                        } else {
                                            MyApp.Yheadnumber--;
                                        }
                                    }
                                }
                            }
                        }

                        if (PoseClassList.SS10(YWangle, MyApp.TPtype,MyApp.Down)) {
                            MyApp.SP = true;

                            if (PoseClassList.DX(pose.getAllPoseLandmarks().get(11).getPosition3D().getZ(),pose.getAllPoseLandmarks().get(12).getPosition3D().getZ())) {

                                MyApp.YXjiaodu=getAngle(pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(24));
                                if (PoseClassList.DX8(getAngle(pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(24)),getAngle(pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(24)))) {
                                    MyApp. Knee = true;

                                } else {

                                }

                            } else {

                                MyApp.YXjiaodu=getAngle(pose.getAllPoseLandmarks().get(13), pose.getAllPoseLandmarks().get(11), pose.getAllPoseLandmarks().get(23));
                                if (PoseClassList.DX8(getAngle(pose.getAllPoseLandmarks().get(13), pose.getAllPoseLandmarks().get(11), pose.getAllPoseLandmarks().get(23)),getAngle(pose.getAllPoseLandmarks().get(13), pose.getAllPoseLandmarks().get(11), pose.getAllPoseLandmarks().get(23)))) {
                                    MyApp. Knee = true;
                                } else {

                                }

                            }

                            if (!MyApp.bend) {
                                Log.d("TAG","Bend");
                            }else if(!hips){
                                if (MyApp. Knee) {
                                    if (Yhead||MyApp.Swing) {
                                        Log.d("TAG","handlder");
                                    } else {
                                        Log.d("TAG","Good");
                                    }
                                } else {
                                    Log.d("TAG","Knee");
                                }
                            }else{
                                Log.d("TAG","hips");

                            }
                        } else if (PoseClassList.SS11(YWangle, MyApp.TPtype,MyApp.YWtype,MyApp.Down)) {
                            Log.d("TAG","flat");
                        } else if (PoseClassList.SS12(MyApp.TPtype,MyApp.QZtype,MyApp.Down,YWangle,YWangle)) {
                            Log.d("TAG","SP");
                        }
                        break;
                }
                break;

            case "C-hes":


                if (PoseClassList.JF(pose.getAllPoseLandmarks().get(30).getPosition3D().getY(),pose.getAllPoseLandmarks().get(24).getPosition3D().getY())) {
                    MyApp.Down = true;
                } else {
                    MyApp.Down = false;
                }

                if (PoseClassList.ZX1(MyApp.Down,pose.getAllPoseLandmarks().get(11).getPosition3D().getZ(),pose.getAllPoseLandmarks().get(12).getPosition3D().getZ())) {
                    YWangle = getAngle(pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(24), pose.getAllPoseLandmarks().get(28));
                    Qtangle = getAngle(pose.getAllPoseLandmarks().get(24), pose.getAllPoseLandmarks().get(26), pose.getAllPoseLandmarks().get(28));
                } else {
                    YWangle = getAngle(pose.getAllPoseLandmarks().get(11), pose.getAllPoseLandmarks().get(23), pose.getAllPoseLandmarks().get(27));
                    Qtangle = getAngle(pose.getAllPoseLandmarks().get(23), pose.getAllPoseLandmarks().get(25), pose.getAllPoseLandmarks().get(27));
                }


                if (PoseClassList.JF1(pose.getAllPoseLandmarks().get(0).getPosition3D().getY(),pose.getAllPoseLandmarks().get(24).getPosition3D().getY(),YWangle,YWangle,MyApp.Down)) {

                    if (PoseClassList.JF2(pose.getAllPoseLandmarks().get(12).getPosition3D().getY(),pose.getAllPoseLandmarks().get(30).getPosition3D().getY(),pose.getAllPoseLandmarks().get(12).getPosition3D().getY(),pose.getAllPoseLandmarks().get(30).getPosition3D().getY())) {
                        MyApp.TPtype = true;
                        MyApp.dis=pose.getAllPoseLandmarks().get(16).getPosition3D().getX();
                        if(PoseClassList.JF3(Qtangle,Qtangle)){
                            MyApp.bend = true;
                        }else{
                            MyApp.bend = false;
                        }
                    }
                }


                if (PoseClassList.ZX1(MyApp.Down,pose.getAllPoseLandmarks().get(11).getPosition3D().getZ() ,pose.getAllPoseLandmarks().get(12).getPosition3D().getZ())) {

                    if(PoseClassList.JF4(pose.getAllPoseLandmarks().get(32).getPosition3D().getY(),pose.getAllPoseLandmarks().get(30).getPosition3D().getY(),(Integer)SharedPreferencesUtils.getParam(MyApp.sContext, "jiaodiC", 5))){
                        MyApp.tOff= true;
                    }else{

                    }

                    TTangel = (pose.getAllPoseLandmarks().get(16).getPosition3D().getY()-pose.getAllPoseLandmarks().get(24).getPosition3D().getY());
                    if (PoseClassList.JF5(MyApp.TPtype,TTangel,(Integer) SharedPreferencesUtils.getParam(MyApp.sContext, "jiaodiY", 13))) {

                        hips = true;
                    } else {

                    }

                    MyApp.dis=pose.getAllPoseLandmarks().get(16).getPosition3D().getX();
                    double armsangle = getAngle(pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(16));
                    MyApp.YWangle=YWangle;


                    if(PoseClassList.JF6(MyApp.dis,MyApp.dis,pose.getAllPoseLandmarks().get(0).getPosition3D().getY(),pose.getAllPoseLandmarks().get(26).getPosition3D().getY(),MyApp.TPtype)){

                        if(PoseClassList.JF7(armsangle,130+((Integer) SharedPreferencesUtils.getParam(MyApp.sContext, "juanfujiaodu", 25)))){
                            MyApp.arms = true;
                        }

                        if (MyApp.bend) {
                            Log.d("TAG","bend");
                        }else if(hips){
                            Log.d("TAG","hips");
                        }else if(MyApp.tOff){
                            Log.d("TAG","tOff");
                        }else if(MyApp.arms){
                            Log.d("TAG","arms");
                        }else{
                            Log.d("TAG","good");
                        }
                    }
                }
                break;


            case "P-up":

                if(PoseClassList.YT(overlay.getImageWidth())){
                    MyApp.pole=80;
                    MyApp.hand=30;
                    MyApp.shanke=15;
                    MyApp.Go=4;
                }else if(PoseClassList.YT1(overlay.getImageWidth())){
                    MyApp.pole=160;
                    MyApp.hand=40;
                    MyApp.shanke=80;
                    MyApp.Go=5;
                }

                MyApp.noX=pose.getAllPoseLandmarks().get(0).getPosition3D().getX();
                MyApp.noY=pose.getAllPoseLandmarks().get(0).getPosition3D().getY();

            case "N1":

                if(true) {
                    float Z = pose.getAllPoseLandmarks().get(19).getPosition3D().getY();
                    float Y = pose.getAllPoseLandmarks().get(20).getPosition3D().getY();

                    MyApp.Youhand=pose.getAllPoseLandmarks().get(20).getPosition3D().getY();
                    MyApp.Zuohand=pose.getAllPoseLandmarks().get(19).getPosition3D().getY();
                    MyApp.Lshoe = pose.getAllPoseLandmarks().get(31).getPosition3D().getX();
                    MyApp.Rshoe = pose.getAllPoseLandmarks().get(32).getPosition3D().getX();

                    MyApp.LS = pose.getAllPoseLandmarks().get(11).getPosition3D().getX();
                    MyApp.RS = pose.getAllPoseLandmarks().get(12).getPosition3D().getX();


                    MyApp.Pos=pose.getAllPoseLandmarks().get(9).getPosition3D().getY();
                    MyApp.armsangle1 = getAngle(pose.getAllPoseLandmarks().get(11), pose.getAllPoseLandmarks().get(13), pose.getAllPoseLandmarks().get(15));
                    MyApp.armsangle2 = getAngle(pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(16));


                    if (PoseClassList.YT3(Z,MyApp.pole,MyApp.hand,Y)) {

                        if (!MyApp.KS) {
                            Log.d("TAG","start");
                        }

                        if(!MyApp.KS){
                            return;
                        }


                        if(MyApp.SM>pose.getAllPoseLandmarks().get(12).getPosition3D().getY()){
                            MyApp.SM=pose.getAllPoseLandmarks().get(12).getPosition3D().getY();
                            MyApp.XM=pose.getAllPoseLandmarks().get(12).getPosition3D().getY();
                        }


                        if(PoseClassList.YT4(MyApp.ZuiDiPos,pose.getAllPoseLandmarks().get(12).getPosition3D().getY())){
                            MyApp.up=true;
                        }
                        if(PoseClassList.YT6(MyApp.up,MyApp.SM,pose.getAllPoseLandmarks().get(12).getPosition3D().getY(),pose.getAllPoseLandmarks().get(9).getPosition3D().getY(),MyApp.pole)){
                            MyApp.down1=true;
                        }


                        if (PoseClassList.YT7(pose.getAllPoseLandmarks().get(5).getPosition3D().getY(),MyApp.pole,pose.getAllPoseLandmarks().get(2).getPosition3D().getY(),MyApp.pole)) {
                            MyApp.Xg = true;
                            if( MyApp.ZuiDiPos==9999){
                                MyApp.ZuiDiPos=pose.getAllPoseLandmarks().get(12).getPosition3D().getY();
                            }else if(MyApp.number==0){
                                MyApp.ZuiDiPos=pose.getAllPoseLandmarks().get(12).getPosition3D().getY();
                            }
                        }

                        if(PoseClassList.YT8(MyApp.Pos,MyApp.pole,MyApp.Go)){
                            MyApp.Go = true;
                            MyApp.upGan=true;
                        }else{
                            MyApp.Go = false;
                        }



                        if(MyApp.Xg&&MyApp.Go){

                            if (MyApp.shanke) {
                                Log.d("TAG","shanke");
                            } else {
                                Log.d("TAG","good");
                            }

                        }else if(PoseClassList.YT9(MyApp.down1,MyApp.up,MyApp.Xg,MyApp.Go)){
                            Log.d("TAG","bad");
                        }

                    } else {
                        if (MyApp.KS) {
                            if(PoseClassList.YT10(Z,MyApp.pole,MyApp.hand,Y)){

                            }else{
                                Log.d("TAG","end");
                            }
                        } else {
                        }
                    }

                }

                break;
            case "N2":

                if(true) {

                    float Z = pose.getAllPoseLandmarks().get(19).getPosition3D().getY();
                    float Y = pose.getAllPoseLandmarks().get(20).getPosition3D().getY();
                    MyApp.Lshoe = pose.getAllPoseLandmarks().get(31).getPosition3D().getX();
                    MyApp.Rshoe = pose.getAllPoseLandmarks().get(32).getPosition3D().getX();
                    MyApp.LS = pose.getAllPoseLandmarks().get(11).getPosition3D().getX();
                    MyApp.RS = pose.getAllPoseLandmarks().get(12).getPosition3D().getX();

                    MyApp.Youhand=pose.getAllPoseLandmarks().get(20).getPosition3D().getY();
                    MyApp.Zuohand=pose.getAllPoseLandmarks().get(19).getPosition3D().getY();
                    MyApp.Pos=pose.getAllPoseLandmarks().get(9).getPosition3D().getY();
                    MyApp.armsangle1 = getAngle(pose.getAllPoseLandmarks().get(11), pose.getAllPoseLandmarks().get(13), pose.getAllPoseLandmarks().get(15));
                    MyApp.armsangle2 = getAngle(pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(16));


                    MyApp.JiaoDiZuo=pose.getAllPoseLandmarks().get(30).getPosition3D().getY();
                    MyApp.JiaoDiYou=pose.getAllPoseLandmarks().get(29).getPosition3D().getY();

                    if (PoseClassList.YT3(Z,MyApp.pole,MyApp.hand,Y)) {
                        if (!MyApp.KS) {
                            Log.d("TAG","start");
                        }


                        if((boolean)SharedPreferencesUtils.getParam(MyApp.sContext, "bddj", true)){

                        }else{

                            if (pose.getAllPoseLandmarks().get(30).getPosition3D().getY() - pose.getAllPoseLandmarks().get(26).getPosition3D().getY() < MyApp.shanke-15 || pose.getAllPoseLandmarks().get(29).getPosition3D().getY() - pose.getAllPoseLandmarks().get(25).getPosition3D().getY() < MyApp.shanke-15) {
                                MyApp.shanke = true;
                            }

                        }


                        if(MyApp.SM>pose.getAllPoseLandmarks().get(12).getPosition3D().getY()){
                            MyApp.SM=pose.getAllPoseLandmarks().get(12).getPosition3D().getY();
                            MyApp.XM=pose.getAllPoseLandmarks().get(12).getPosition3D().getY();
                            if (PoseClassList.YT12(MyApp.Xg,pose.getAllPoseLandmarks().get(26).getPosition3D().getY(),pose.getAllPoseLandmarks().get(24).getPosition3D().getY(),MyApp.shanke,pose.getAllPoseLandmarks().get(25).getPosition3D().getY(),pose.getAllPoseLandmarks().get(23).getPosition3D().getY())) {
                                MyApp.shanke = true;
                            }
                        }


                        if(PoseClassList.YT11(MyApp.up,MyApp.XM,pose.getAllPoseLandmarks().get(12).getPosition3D().getY(),pose.getAllPoseLandmarks().get(11).getPosition3D().getY())){
                            MyApp.XM=pose.getAllPoseLandmarks().get(12).getPosition3D().getY();
                            if (PoseClassList.YT12(MyApp.Xg,pose.getAllPoseLandmarks().get(26).getPosition3D().getY(),pose.getAllPoseLandmarks().get(24).getPosition3D().getY(),MyApp.shanke,pose.getAllPoseLandmarks().get(25).getPosition3D().getY(),pose.getAllPoseLandmarks().get(23).getPosition3D().getY())) {
                                MyApp.shanke = true;
                            }
                        }

                        if(PoseClassList.YT4(MyApp.ZuiDiPos,pose.getAllPoseLandmarks().get(12).getPosition3D().getY())){
                            MyApp.up=true;
                            if (PoseClassList.YT12(MyApp.Xg,pose.getAllPoseLandmarks().get(26).getPosition3D().getY(),pose.getAllPoseLandmarks().get(24).getPosition3D().getY(),MyApp.shanke,pose.getAllPoseLandmarks().get(25).getPosition3D().getY(),pose.getAllPoseLandmarks().get(23).getPosition3D().getY())) {
                                MyApp.shanke = true;
                            }
                        }
                        if(PoseClassList.YT61(MyApp.up,MyApp.SM,pose.getAllPoseLandmarks().get(12).getPosition3D().getY(),pose.getAllPoseLandmarks().get(9).getPosition3D().getY(),MyApp.pole)){
                            MyApp.down1=true;
                        }


                        if (PoseClassList.YT13(getAngle(pose.getAllPoseLandmarks().get(16), pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(12)),pose.getAllPoseLandmarks().get(10).getPosition3D().getY(),pose.getAllPoseLandmarks().get(14).getPosition3D().getY(),getAngle(pose.getAllPoseLandmarks().get(15), pose.getAllPoseLandmarks().get(13), pose.getAllPoseLandmarks().get(11)),pose.getAllPoseLandmarks().get(9).getPosition3D().getY(),pose.getAllPoseLandmarks().get(13).getPosition3D().getY())) {
                            MyApp.Xg = true;
                            if( MyApp.ZuiDiPos==9999){
                                MyApp.ZuiDiPos=pose.getAllPoseLandmarks().get(12).getPosition3D().getY();
                            }else if(MyApp.number==0){
                                MyApp.ZuiDiPos=pose.getAllPoseLandmarks().get(12).getPosition3D().getY();
                            }
                        }
                        if(PoseClassList.YT8(MyApp.Pos,MyApp.pole,MyApp.Go)){
                            MyApp.Go = true;
                            MyApp.upGan=true;
                            if (PoseClassList.YT12(MyApp.Xg,pose.getAllPoseLandmarks().get(26).getPosition3D().getY(),pose.getAllPoseLandmarks().get(24).getPosition3D().getY(),MyApp.shanke,pose.getAllPoseLandmarks().get(25).getPosition3D().getY(),pose.getAllPoseLandmarks().get(23).getPosition3D().getY())) {
                                MyApp.shanke = true;
                            }
                        }else{
                            MyApp.Go = false;
                        }



                        if(MyApp.Xg&&MyApp.Go){

                            if (MyApp.shanke) {
                                Log.d("TAG","shanke");
                            } else {
                                Log.d("TAG","good");
                            }

                        }else if(PoseClassList.YT14(MyApp.upGan,MyApp.Xg,MyApp.up,MyApp.Go,MyApp.down1,getAngle(pose.getAllPoseLandmarks().get(16), pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(12)),pose.getAllPoseLandmarks().get(10).getPosition3D().getY(),pose.getAllPoseLandmarks().get(14).getPosition3D().getY(),MyApp.upGan,MyApp.Xg,MyApp.up,MyApp.Go,MyApp.down1,getAngle(pose.getAllPoseLandmarks().get(15), pose.getAllPoseLandmarks().get(13), pose.getAllPoseLandmarks().get(11)),pose.getAllPoseLandmarks().get(9).getPosition3D().getY(),pose.getAllPoseLandmarks().get(13).getPosition3D().getY())){
                            Log.d("TAG","bad");


                        }else if( PoseClassList.YT15(MyApp.upGan,MyApp.Xg,MyApp.up,MyApp.Go,MyApp.down1,getAngle(pose.getAllPoseLandmarks().get(16), pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(12)))){

                        }else if(PoseClassList.YT9(MyApp.down1,MyApp.up,MyApp.Xg,MyApp.Go)){
                            Log.d("TAG","bad");

                        }


                        if((boolean)SharedPreferencesUtils.getParam(MyApp.sContext, "jcgz", false)) {
                            if (MyApp.Lshoe != 0 && MyApp.Rshoe != 0 && MyApp.LS != 0 && MyApp.RS != 0) {
                                if (PoseClassList.YT16(MyApp.Lshoe,MyApp.Rshoe,MyApp.LS,MyApp.RS)) {
                                    if (MyApp.KS) {
                                        Log.d("TAG","end");
                                    }
                                }
                            }
                        }
                    } else {
                        if (MyApp.KS) {
                            if(PoseClassList.YT10(Z,MyApp.pole,MyApp.hand,Y)){

                            }else{
                                Log.d("TAG","end");
                            }
                        } else {

                        }
                    }
                }

                break;

            case "N3":

                if(true) {
                    MyApp.shankeFuDu=pose.getAllPoseLandmarks().get(32).getPosition3D().getY() - pose.getAllPoseLandmarks().get(26).getPosition3D().getY();

                    float Z = pose.getAllPoseLandmarks().get(19).getPosition3D().getY();
                    float Y = pose.getAllPoseLandmarks().get(20).getPosition3D().getY();

                    MyApp.Lshoe = pose.getAllPoseLandmarks().get(31).getPosition3D().getX();
                    MyApp.Rshoe = pose.getAllPoseLandmarks().get(32).getPosition3D().getX();

                    MyApp.LS = pose.getAllPoseLandmarks().get(11).getPosition3D().getX();
                    MyApp.RS = pose.getAllPoseLandmarks().get(12).getPosition3D().getX();

                    MyApp.Youhand=pose.getAllPoseLandmarks().get(20).getPosition3D().getY();
                    MyApp.Zuohand=pose.getAllPoseLandmarks().get(19).getPosition3D().getY();
                    MyApp.Pos=pose.getAllPoseLandmarks().get(9).getPosition3D().getY();
                    MyApp.armsangle1 = getAngle(pose.getAllPoseLandmarks().get(11), pose.getAllPoseLandmarks().get(13), pose.getAllPoseLandmarks().get(15));
                    MyApp.armsangle2 = getAngle(pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(16));


                    MyApp.JiaoDiZuo=pose.getAllPoseLandmarks().get(30).getPosition3D().getY();
                    MyApp.JiaoDiYou=pose.getAllPoseLandmarks().get(29).getPosition3D().getY();


                    if(MyApp.noX>240-80||MyApp.noX<80){
                        Log.d("TAG","end");
                    }


                    if (PoseClassList.YT3(Z,MyApp.pole,MyApp.hand,Y)) {
                        if (!MyApp.KS) {
                            Log.d("TAG","start");

                        }


                        MyApp. Kneean=new Date(System.currentTimeMillis());



                        if((boolean)SharedPreferencesUtils.getParam(MyApp.sContext, "bddj", true)){

                        }else{


                            if ( PoseClassList.YT121(pose.getAllPoseLandmarks().get(32).getPosition3D().getY(),pose.getAllPoseLandmarks().get(26).getPosition3D().getY(), MyApp.shanke,pose.getAllPoseLandmarks().get(26).getPosition3D().getY(),pose.getAllPoseLandmarks().get(24).getPosition3D().getY(),pose.getAllPoseLandmarks().get(31).getPosition3D().getY(), pose.getAllPoseLandmarks().get(25).getPosition3D().getY(), MyApp.shanke,pose.getAllPoseLandmarks().get(25).getPosition3D().getY(),pose.getAllPoseLandmarks().get(23).getPosition3D().getY())) {
                                MyApp.shanke = true;
                            }


                            if (PoseClassList.YT12(MyApp.Xg,pose.getAllPoseLandmarks().get(26).getPosition3D().getY(),pose.getAllPoseLandmarks().get(24).getPosition3D().getY(),MyApp.shanke,pose.getAllPoseLandmarks().get(25).getPosition3D().getY(),pose.getAllPoseLandmarks().get(23).getPosition3D().getY())) {
                                MyApp.shanke = true;
                            }

                        }


                        if(MyApp.SM>pose.getAllPoseLandmarks().get(12).getPosition3D().getY()){
                            MyApp.SM=pose.getAllPoseLandmarks().get(12).getPosition3D().getY();
                            MyApp.XM=pose.getAllPoseLandmarks().get(12).getPosition3D().getY();

                        }

                        if(PoseClassList.YT11(MyApp.up,MyApp.XM,pose.getAllPoseLandmarks().get(12).getPosition3D().getY(),pose.getAllPoseLandmarks().get(11).getPosition3D().getY())){
                            MyApp.XM=pose.getAllPoseLandmarks().get(12).getPosition3D().getY();
                        }


                        if(PoseClassList.YT17(pose.getAllPoseLandmarks().get(5).getPosition3D().getY(),pose.getAllPoseLandmarks().get(14).getPosition3D().getY(),pose.getAllPoseLandmarks().get(2).getPosition3D().getY(),pose.getAllPoseLandmarks().get(13).getPosition3D().getY())){
                            MyApp.up=true;
                        }
                        if(PoseClassList.YT61(MyApp.up,MyApp.SM,pose.getAllPoseLandmarks().get(12).getPosition3D().getY(),pose.getAllPoseLandmarks().get(9).getPosition3D().getY(),MyApp.pole)){
                            MyApp.down1=true;
                        }

                        if (PoseClassList.YT131(getAngle(pose.getAllPoseLandmarks().get(16), pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(12)),pose.getAllPoseLandmarks().get(5).getPosition3D().getY(),pose.getAllPoseLandmarks().get(14).getPosition3D().getY(),getAngle(pose.getAllPoseLandmarks().get(15), pose.getAllPoseLandmarks().get(13), pose.getAllPoseLandmarks().get(11)),pose.getAllPoseLandmarks().get(2).getPosition3D().getY(),pose.getAllPoseLandmarks().get(13).getPosition3D().getY())) {
                            MyApp.Xg = true;
                        }

                        if(PoseClassList.YT8(MyApp.Pos,MyApp.pole,MyApp.Go)){
                            MyApp.Go = true;
                            MyApp.upGan=true;
                        }else{
                            MyApp.Go = false;
                        }


                        if(MyApp.Xg&&MyApp.Go){
                            if (MyApp.shanke) {
                                Log.d("TAG","shanke");
                            } else {
                                Log.d("TAG","good");
                            }

                        }else  if(PoseClassList.YT141(MyApp.upGan,MyApp.Xg,MyApp.up,MyApp.Go,MyApp.down1,getAngle(pose.getAllPoseLandmarks().get(16), pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(12)),pose.getAllPoseLandmarks().get(5).getPosition3D().getY(),pose.getAllPoseLandmarks().get(14).getPosition3D().getY(),
                                MyApp.upGan,MyApp.Xg,MyApp.up,MyApp.Go,MyApp.down1,getAngle(pose.getAllPoseLandmarks().get(15), pose.getAllPoseLandmarks().get(13), pose.getAllPoseLandmarks().get(11)),pose.getAllPoseLandmarks().get(2).getPosition3D().getY(),pose.getAllPoseLandmarks().get(13).getPosition3D().getY())){
                            Log.d("TAG","bad");

                        }else if(PoseClassList.YT151(MyApp.upGan,MyApp.Xg,MyApp.up,MyApp.Go,MyApp.down1,getAngle(pose.getAllPoseLandmarks().get(16), pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(12)))){

                        }else if(PoseClassList.YT9(MyApp.down1,MyApp.up,MyApp.Xg,MyApp.Go)){
                            Log.d("TAG","bad");
                        }


                        if((boolean)SharedPreferencesUtils.getParam(MyApp.sContext, "jcgz", false)) {
                            if (MyApp.Lshoe != 0 && MyApp.Rshoe != 0 && MyApp.LS != 0 && MyApp.RS != 0) {
                                if (PoseClassList.YT16(MyApp.Lshoe,MyApp.Rshoe,MyApp.LS,MyApp.RS)) {
                                    if (MyApp.KS) {
                                        Log.d("TAG","end");
                                    }
                                }
                            }
                        }

                    } else {
                        if (MyApp.KS) {
                            if(PoseClassList.YT10(Z,MyApp.pole,MyApp.hand,Y)){

                            }else{
                                Log.d("TAG","end");
                            }


                        } else {
                        }
                    }
                }

                break;
            break;

            case "Cas":

                MyApp.GanZipos=80;
                MyApp.shou=30;
                MyApp.baidong=25;
                MyApp.guogan=3;

                MyApp.Youhand=pose.getAllPoseLandmarks().get(20).getPosition3D().getY();
                MyApp.Zuohand=pose.getAllPoseLandmarks().get(19).getPosition3D().getY();
                MyApp.Pos=pose.getAllPoseLandmarks().get(9).getPosition3D().getY();
                MyApp.Pos1=pose.getAllPoseLandmarks().get(10).getPosition3D().getY();
                if(pose.getAllPoseLandmarks().get(19).getInFrameLikelihood()>0.8&&pose.getAllPoseLandmarks().get(20).getInFrameLikelihood()>0.8){
                    if (PoseClassList.QB1(zuo,MyApp.GanZipos,MyApp.shou,you,MyApp.pos)) {
                        if (!MyApp.KS) {
                            Log.d("TAG","start");
                        }
                    }else{
                        if (MyApp.KS) {
                            Log.d("TAG","end");
                        } else {

                        }
                    }
                }
                break;

            case "FPU":
            case "N":
                if (pose.getAllPoseLandmarks().get(15).getPosition3D().getZ() > pose.getAllPoseLandmarks().get(16).getPosition3D().getZ()) {
                    if (pose.getAllPoseLandmarks().get(12).getInFrameLikelihood() > 0.9 && pose.getAllPoseLandmarks().get(14).getInFrameLikelihood() > 0.9&& pose.getAllPoseLandmarks().get(16).getInFrameLikelihood() > 0.9&& pose.getAllPoseLandmarks().get(32).getInFrameLikelihood() > 0.9) {

                        if(pose.getAllPoseLandmarks().get(12).getPosition3D().getY()-pose.getAllPoseLandmarks().get(14).getPosition3D().getY()>5){
                            MyApp.PD=true;
                        }
                        Qarmsangle
                        MyApp.Qarmsangle = getAngle(pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(16));
                        MyApp.Parmsangle = getAngle(pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(24), pose.getAllPoseLandmarks().get(26));
                        MyApp.Garmsangle = getAngle(pose.getAllPoseLandmarks().get(24), pose.getAllPoseLandmarks().get(26), pose.getAllPoseLandmarks().get(28));
                        MyApp.Garmsangle1=160;
                        MyApp.pos15 = pose.getAllPoseLandmarks().get(16).getPosition3D().getY();
                        MyApp.pos25 = pose.getAllPoseLandmarks().get(26).getPosition3D().getY();
                        MyApp.pos31 = pose.getAllPoseLandmarks().get(32).getPosition3D().getY();
                        MyApp.pos11 = pose.getAllPoseLandmarks().get(12).getPosition3D().getY();
                        MyApp.pos13 = pose.getAllPoseLandmarks().get(14).getPosition3D().getY();


                        if (PoseClassList.FC1(pose.getAllPoseLandmarks().get(12).getPosition3D().getY(),pose.getAllPoseLandmarks().get(14).getPosition3D().getY(),40)) {
                            MyApp.MD = true;
                            MyApp.UD = false;
                            MyApp.AX=false;
                        }
                        if (PoseClassList.FC2(pose.getAllPoseLandmarks().get(12).getPosition3D().getY(),pose.getAllPoseLandmarks().get(14).getPosition3D().getY(),25,14)) {
                            Log.d("TAG","down");
                        }
                        if (PoseClassList.FC3(MyApp.MD, getAngle(pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(16)),145) ) {
                            Log.d("TAG","up");
                        }
                    }
                    if (PoseClassList.FC4(MyApp.MD,getAngle(pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(16)),165,pose.getAllPoseLandmarks().get(14).getPosition3D().getY(),pose.getAllPoseLandmarks().get(12).getPosition3D().getY(),30)) {
                        Log.d("TAG","up");
                    }

                }
                if(PoseClassList.FC6(MyApp.MD,MyApp.UD,MyApp.AX,MyApp.QX)){
                    Log.d("TAG","bad");
                }else if(PoseClassList.FC7(MyApp.QX,MyApp.AX,MyApp.MD,MyApp.UD)){
                    Log.d("TAG","good");
                }
                if(MyApp.PD){
                    Log.d("TAG","end");
                }
                break;

            case "PB":
                if(overlay.getImageWidth()==240){
                    MyApp.Gzpos=160;
                }
                MyApp.Qarmsangle = getAngle(pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(16));
                MyApp.Parmsangle = getAngle(pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(24), pose.getAllPoseLandmarks().get(26));
                MyApp.Garmsangle = getAngle(pose.getAllPoseLandmarks().get(24), pose.getAllPoseLandmarks().get(26), pose.getAllPoseLandmarks().get(28));
                MyApp.Garmsangle1=160;
                MyApp.pos15 = pose.getAllPoseLandmarks().get(16).getPosition3D().getY();
                MyApp.pos25 = pose.getAllPoseLandmarks().get(26).getPosition3D().getY();
                MyApp.pos31 = pose.getAllPoseLandmarks().get(32).getPosition3D().getY();
                MyApp.pos11 = pose.getAllPoseLandmarks().get(12).getPosition3D().getY();
                MyApp.pos13 = pose.getAllPoseLandmarks().get(14).getPosition3D().getY();


                if (PoseClassList.FC1(pose.getAllPoseLandmarks().get(12).getPosition3D().getY(),pose.getAllPoseLandmarks().get(14).getPosition3D().getY(),40)) {
                    MyApp.MD = true;
                    MyApp.UD = false;
                    MyApp.AX=false;
                }
                if (PoseClassList.FC2(pose.getAllPoseLandmarks().get(12).getPosition3D().getY(),pose.getAllPoseLandmarks().get(14).getPosition3D().getY(),25,14)) {
                    MyApp.QX = true;
                    MyApp.UD = false;
                    MyApp.AX=false;
                }
                if (PoseClassList.FC3(MyApp.MD, getAngle(pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(16)),145) ) {
                    MyApp.UD = true;
                }
                if (PoseClassList.FC4(MyApp.MD,getAngle(pose.getAllPoseLandmarks().get(12), pose.getAllPoseLandmarks().get(14), pose.getAllPoseLandmarks().get(16)),165,pose.getAllPoseLandmarks().get(14).getPosition3D().getY(),pose.getAllPoseLandmarks().get(12).getPosition3D().getY(),30)) {
                    MyApp.AX = true;
                }

                if(PoseClassList.FC6(MyApp.MD,MyApp.UD,MyApp.AX,MyApp.QX)){
                    Log.d("TAG","bad");
                }
                if(PoseClassList.FC7(MyApp.QX,MyApp.AX,MyApp.MD,MyApp.UD)){
                    Log.d("TAG","good");
                }
                break;

        }
        drawLine(canvas, nose, lefyEyeInner, whitePaint);
        drawLine(canvas, lefyEyeInner, lefyEye, whitePaint);
        drawLine(canvas, lefyEye, leftEyeOuter, whitePaint);
        drawLine(canvas, leftEyeOuter, leftEar, whitePaint);
        drawLine(canvas, nose, rightEyeInner, whitePaint);
        drawLine(canvas, rightEyeInner, rightEye, whitePaint);
        drawLine(canvas, rightEye, rightEyeOuter, whitePaint);
        drawLine(canvas, rightEyeOuter, rightEar, whitePaint);
        drawLine(canvas, leftMouth, rightMouth, whitePaint);

        drawLine(canvas, lefthandlder, righthandlder, whitePaint);
        drawLine(canvas, leftHip, rightHip, whitePaint);

        drawLine(canvas, lefthandlder, leftElbow, leftPaint);
        drawLine(canvas, leftElbow, leftWrist, leftPaint);
        drawLine(canvas, lefthandlder, leftHip, leftPaint);
        drawLine(canvas, leftHip, leftKnee, leftPaint);
        drawLine(canvas, leftKnee, leftAnkle, leftPaint);
        drawLine(canvas, leftWrist, leftThumb, leftPaint);
        drawLine(canvas, leftWrist, leftPinky, leftPaint);
        drawLine(canvas, leftWrist, leftIndex, leftPaint);
        drawLine(canvas, leftIndex, leftPinky, leftPaint);
        drawLine(canvas, leftAnkle, leftHeel, leftPaint);
        drawLine(canvas, leftHeel, leftFootIndex, leftPaint);

        drawLine(canvas, righthandlder, rightElbow, rightPaint);
        drawLine(canvas, rightElbow, rightWrist, rightPaint);
        drawLine(canvas, righthandlder, rightHip, rightPaint);
        drawLine(canvas, rightHip, rightKnee, rightPaint);
        drawLine(canvas, rightKnee, rightAnkle, rightPaint);
        drawLine(canvas, rightWrist, rightThumb, rightPaint);
        drawLine(canvas, rightWrist, rightPinky, rightPaint);
        drawLine(canvas, rightWrist, rightIndex, rightPaint);
        drawLine(canvas, rightIndex, rightPinky, rightPaint);
        drawLine(canvas, rightAnkle, rightHeel, rightPaint);
        drawLine(canvas, rightHeel, rightFootIndex, rightPaint);

        if (showInFrameLikelihood) {
            for (PoseLandmark landmark : landmarks) {
                canvas.drawText(
                        String.format(Locale.US, "%.2f", landmark.getInFrameLikelihood()),
                        translateX(landmark.getPosition().x),
                        translateY(landmark.getPosition().y),
                        whitePaint);
            }
        }
    }

    void drawPoint(Canvas canvas, PoseLandmark landmark, Paint paint) {
        PointF3D point = landmark.getPosition3D();
        maybeUpdatePaintColor(paint, canvas, point.getZ());
        canvas.drawCircle(translateX(point.getX()), translateY(point.getY()), DOT_RADIUS, paint);
    }

    void drawLine(Canvas canvas, PoseLandmark startLandmark, PoseLandmark endLandmark, Paint paint) {
        PointF3D start = startLandmark.getPosition3D();
        PointF3D end = endLandmark.getPosition3D();

        float avgZInImagePixel = (start.getZ() + end.getZ()) / 2;
        maybeUpdatePaintColor(paint, canvas, avgZInImagePixel);

        canvas.drawLine(
                translateX(start.getX()),
                translateY(start.getY()),
                translateX(end.getX()),
                translateY(end.getY()),
                paint);
    }

    private void maybeUpdatePaintColor(Paint paint, Canvas canvas, float zInImagePixel) {
        if (!visualizeZ) {
            return;
        }

        float zLowerBoundInScreenPixel;
        float zUpperBoundInScreenPixel;

        if (rescaleZForVisualization) {
            zLowerBoundInScreenPixel = min(-0.001f, scale(zMin));
            zUpperBoundInScreenPixel = max(0.001f, scale(zMax));
        } else {

            float defaultRangeFactor = 1f;
            zLowerBoundInScreenPixel = -defaultRangeFactor * canvas.getWidth();
            zUpperBoundInScreenPixel = defaultRangeFactor * canvas.getWidth();
        }

        float zInScreenPixel = scale(zInImagePixel);

        if (zInScreenPixel < 0) {
            int v = (int) (zInScreenPixel / zLowerBoundInScreenPixel * 255);
            v = Ints.constrainToRange(v, 0, 255);
            paint.setARGB(255, 255, 255 - v, 255 - v);
        } else {

            int v = (int) (zInScreenPixel / zUpperBoundInScreenPixel * 255);
            v = Ints.constrainToRange(v, 0, 255);
            paint.setARGB(255, 255 - v, 255 - v, 255);
        }
    }

    public  double getAngle(PoseLandmark firstPoint, PoseLandmark midPoint, PoseLandmark lastPoint) {
        double result =
                Math.toDegrees(
                        atan2(lastPoint.getPosition().y - midPoint.getPosition().y,
                                lastPoint.getPosition().x - midPoint.getPosition().x)
                                - atan2(firstPoint.getPosition().y - midPoint.getPosition().y,
                                firstPoint.getPosition().x - midPoint.getPosition().x));
        result = Math.abs(result);
        if (result > 180) {
            result = (360.0 - result);
        }
        return result;
    }

}
