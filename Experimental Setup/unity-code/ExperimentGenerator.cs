using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems; // <-- new

// experiment currently in CSV file, each obj with 3 repeats 
// add the UXF namespace
using UXF;

public class ExperimentGenerator : MonoBehaviour
{     

    private GameObject stim;
    public static bool clicked = false; 

    void Start() // <-- new
    {
        // sets stim cross active
        transform.GetChild(0).gameObject.SetActive(true);
    }

    // generate the blocks and trials for the session.
    // the session is passed as an argument by the event call.
    public void Generate(Session session)
    {
        // generate a single block with numTrials trials.
        // int numChildren = stim.transform.childCount();
        // currently 53 trials as of 6/28 4:00 pm 
        int numTrials = 53;
        Block block = session.CreateBlock(numTrials);
        // transform.GetChild(0).gameObject.SetActive(true); 
    }

}