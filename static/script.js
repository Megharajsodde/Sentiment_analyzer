// here write a script of the predict function
async function predict()
{
    const text = document.getElementById("text").value;

    const response = await fetch("/predict",
    {
        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            text: text
        })
    });

    const data = await response.json();

    document.getElementById("result").innerText =
        "Prediction: " + data.result;
}