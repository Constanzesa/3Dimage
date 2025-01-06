using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UXF;
using LSL;

// script for rotating object 

public class Rotate : MonoBehaviour
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
        transform.rotation = Quaternion.Euler(0, 180, 0);
        string name = trial.settings.GetString("object"); 
        Transform obj = transform.Find(name); 

        obj.transform.gameObject.SetActive(true);
        if (outlet != null)
            {
                sample[0] = "StimStart " + obj;
                // Debug.Log(sample[0]);
                outlet.push_sample(sample);
            }
        Invoke("EndAndPrepare", 8);
        speed = 45f; // degrees per second, 8 secs to rotate 360 degrees
        // Invoke("EndAndPrepare", 8);
    }

    // Update is called once per frame
    void Update()
    {
        transform.Rotate(Vector3.up * speed * Time.deltaTime);
    }

    void EndAndPrepare()
        {
            transform.rotation = Quaternion.Euler(0, 180, 0);
            speed = 0f; 
            string name = Session.instance.CurrentTrial.settings.GetString("object"); 
            Transform obj = transform.Find(name); 

            obj.transform.gameObject.SetActive(false);
            if (outlet != null)
            {
                sample[0] = "StimEnd " + obj;
                // Debug.Log(sample[0]);
                outlet.push_sample(sample);
            }
            

            Session.instance.CurrentTrial.End();
            // if current trial is last trial in experiment 
            if (Session.instance.CurrentTrial == Session.instance.LastTrial)
            {
                Invoke("WaitToEnd", 2);
            }

            // if current trial is last trial in block
            if (Session.instance.CurrentTrial == Session.instance.CurrentBlock.lastTrial)
            {
                Invoke("BlockEnd", 2);
            } else {
                Invoke("BeginNext", 2);
            }
        }
           

        void BeginNext()
        {            
            Session.instance.BeginNextTrial();
        }

        void BlockEnd() {
            transform.Find("stimStart").transform.gameObject.SetActive(true);
        }

        void WaitToEnd() {
            Session.instance.End(); // end current session
        }
}
