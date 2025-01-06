using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UXF; 

public class TrialBegin : MonoBehaviour
{
    void Update() 
    {
        if (Input.GetKey("space"))
        {
            Session.instance.FirstTrial.Begin();
        }
       
    }
}
