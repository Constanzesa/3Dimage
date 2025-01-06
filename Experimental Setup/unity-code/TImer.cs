using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UXF;

public class Timer : MonoBehaviour
{
    public Session session;
    
    public void BeginCountdown()
    {
        StartCoroutine(Countdown());
    }

    public void StopCountdown()
    {
        StopAllCoroutines();
    }

    IEnumerator Countdown()
    {
        yield return new WaitForSeconds(2f);
        session.EndCurrentTrial();
    }
}
