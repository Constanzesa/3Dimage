using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UXF;
using LSL;

// script for static display

public class Static : MonoBehaviour
{
    string StreamName = "LSLMarkersInletStreamName2";
                
    string StreamType = "Markers";
    private StreamOutlet outlet;
    private string[] sample = {""};

    public float speed = 0f; 

    public void Start()
    {
        var hash = new Hash128();
        hash.Append(StreamName);
        hash.Append(StreamType);
        hash.Append(gameObject.GetInstanceID());
        StreamInfo streamInfo = new StreamInfo(StreamName, StreamType, 1, LSL.LSL.IRREGULAR_RATE,channel_format_t.cf_string, hash.ToString());
        // StreamInfo streamInfo = new StreamInfo(StreamName, StreamType, LSL.LSL.IRREGULAR_RATE,channel_format='int32', hash.ToString());
        outlet = new StreamOutlet(streamInfo);

        for (int i = 0; i < transform.childCount; i++) {
            transform.GetChild(i).gameObject.SetActive(false);
        }
    }

    public void trialBegin(Trial trial)
    {
        string name = trial.settings.GetString("object"); 
        transform.rotation = Quaternion.Euler(0, 180, 0);

        // Part 1: Object Imagining)
        if(Session.instance.currentBlockNum == 1) {
            Transform obj = transform.Find(name); 
            obj.transform.gameObject.SetActive(true);
            // push to LSL 
            if (outlet != null)
            {
                sample[0] = "Part 1 StimStart " + obj;
                // Debug.Log(sample[0]);
                outlet.push_sample(sample);
            }
            Invoke("Imagining", 8);

            speed = 45f;
        }
        // Part 2: Text Imagining
        else if (Session.instance.currentBlockNum == 2) {
            Transform obj = transform.Find(name); 
            obj.transform.gameObject.SetActive(true);
            if (outlet != null)
            {
                sample[0] = "Part 2 StimStart " + obj;
                // Debug.Log(sample[0]);
                outlet.push_sample(sample);
            }
            Invoke("Imagining", 2);
            speed = 0f;
        }

        // Part 3: MultiModal
        else if (Session.instance.currentBlockNum == 3 || Session.instance.currentBlockNum == 4) {
            Transform obj = transform.Find(name); 
            obj.transform.gameObject.SetActive(true);
            if (outlet != null)
            {
                sample[0] = "Part 3 StimStart " + obj;
                // Debug.Log(sample[0]);
                outlet.push_sample(sample);
            }

            Invoke("EndAndPrepare", 8);
            if (name.Contains("Text")) {
                speed = 0f;
            } else {
                speed = 45f; 
            }
        }
    }

    // update is called once per frame 
    void Update()
    {
        transform.Rotate(Vector3.up * speed * Time.deltaTime);
    }

    void Imagining() {
        speed = 0f;
        Transform obj = transform.Find(Session.instance.CurrentTrial.settings.GetString("object"));
        obj.transform.gameObject.SetActive(false);
        if (outlet != null)
            {
                sample[0] = "Imagination Start " + obj;
                // Debug.Log(sample[0]);
                outlet.push_sample(sample);
            }
        // transform.Find("square").transform.gameObject.SetActive(true);

        if(Session.instance.currentBlockNum == 1) {
            Invoke("EndAndPrepare", 8);
        }
        else if(Session.instance.currentBlockNum == 2) {
            Invoke("EndAndPrepare", 2);
        }
    }

    void EndAndPrepare()
        {
            speed = 0f;
            Transform obj = transform.Find(Session.instance.CurrentTrial.settings.GetString("object"));
            
            if (Session.instance.currentBlockNum == 1 || Session.instance.currentBlockNum == 2) {
                if (outlet != null)
                {
                    sample[0] = "ImaginationEnd " + obj;
                    // Debug.Log(sample[0]);
                    outlet.push_sample(sample);
                }
            }
            
            // blacks out obj for block 3 and 4
            if (Session.instance.currentBlockNum == 3 || Session.instance.currentBlockNum == 4) {
                obj.transform.gameObject.SetActive(false);
                if (outlet != null)
                {
                    sample[0] = "StimEnd " + obj;
                    // Debug.Log(sample[0]);
                    outlet.push_sample(sample);
                }
            }
            transform.Find("restText").transform.gameObject.SetActive(true);


            // ends current trial 
            Session.instance.CurrentTrial.End();

            // if current trial is last trial in experiment 
            if (Session.instance.CurrentTrial == Session.instance.LastTrial)
            {
                Invoke("WaitToEnd", 2);
            }

            if (Session.instance.currentBlockNum == 1) {
                // if current trial is last trial in block
                if (Session.instance.CurrentTrial == Session.instance.CurrentBlock.lastTrial)
                {
                    Invoke("BlockEnd", 4);
                } else {
                    Invoke("BeginNext", 4);
                }
            }
            else {
                if (Session.instance.CurrentTrial == Session.instance.CurrentBlock.lastTrial)
                {
                    Invoke("BlockEnd", 2);
                } else {
                    Invoke("BeginNext", 2);
                }
            }
            
        }

        void BeginNext()
        {            
            Session.instance.BeginNextTrial();
        }

        void BlockEnd() {
            transform.Find("restText").transform.gameObject.SetActive(false);

            if (Session.instance.currentBlockNum == 1) {
                transform.Find("instruct1").transform.gameObject.SetActive(true);
            }
            if (Session.instance.currentBlockNum == 2) {
                transform.Find("instruct2").transform.gameObject.SetActive(true);
            }
            if (Session.instance.currentBlockNum == 3) {
                transform.Find("instruct2").transform.gameObject.SetActive(true);
            }
            
        }

        void WaitToEnd() {
            Session.instance.End(); // end current session
        }
}
