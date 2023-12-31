<?php
session_start();
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Last update: 21.7.2019
//
// Thanks to Amir Dezfouli and Tsahi Asher for their help in improving this code.
//
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

$biased_side = 'biased_side';
$unbiased_side = 'unbiased_side';

/////////////////////////////////////////////////////////////////////////////////
// Get choice in current trial
/////////////////////////////////////////////////////////////////////////////////
$is_biased_side = $_SESSION[$biased_side]==$_GET["side"];
$is_biased_choice = $is_biased_side ? 'True' : 'False';

/////////////////////////////////////////////////////////////////////////////////
// Allocate rewards to next trial
/////////////////////////////////////////////////////////////////////////////////
$result_string = "";
/////////////////////////////////////////////////////////////////////////////
// DYNAMIC Allocation of rewards to next trial
/////////////////////////////////////////////////////////////////////////////
$current_biased_reward = $current_unbiased_reward = null;

// The name of the dynamic schedule should be set in the session parameter, e.g.:
$_SESSION['schedule_name'] = "nadav_lederman";
$_SESSION['schedule_type'] = "DYNAMIC";
// The name should refer to a python file that exists in sequences/dynamic folder
// A good place for this definition is at main.php
if ($_SESSION['schedule_type'] == "DYNAMIC") { //Game is dynamic
    $run_python_command = 'python ../sequences/dynamic/'. $_SESSION['schedule_name'] . '.py '
        . json_encode($_SESSION['bias_rewards']) . ' '
        . json_encode($_SESSION['unbias_rewards']) . ' '
        . json_encode($_SESSION['is_bias_choice']) . ' '
        . json_encode($_SESSION['user_id']);
    $result_string = exec($run_python_command);
    $result_array = explode(", ", $result_string);
    $current_biased_reward = $result_array[0];
    $current_unbiased_reward = $result_array[1];
}
/////////////////////////////////////////////////////////////////////////////
// STATIC - Allocation of rewards to next trial
/////////////////////////////////////////////////////////////////////////////
// else { // Game is static
//     include '../sequences/static/' . $_SESSION['schedule_name'] . '.php';
//     $current_biased_reward = $biased_rewards[$_SESSION['trial_number']];
//     $current_unbiased_reward = $unbiased_rewards[$_SESSION['trial_number']];
// }
if ($is_biased_side) { // If current choice was of the biased side
    $current_reward = $current_biased_reward;
    $current_unobserved_reward = $current_unbiased_reward;
}
else {
    $current_reward = $current_unbiased_reward;
    $current_unobserved_reward = $current_biased_reward;
}
array_push($_SESSION['bias_rewards'], $current_biased_reward);
array_push($_SESSION['unbias_rewards'], $current_unbiased_reward);
array_push($_SESSION['is_bias_choice'], $is_biased_choice);

/////////////////////////////////////////////////////////////////////////////////
// Write current trial's data
/////////////////////////////////////////////////////////////////////////////////
function remove_last_char_from_file($file_name){
    $fh = fopen($file_name, 'r+') or die("can't open file");
    $stat = fstat($fh);
    ftruncate($fh, $stat['size']-1);
    fclose($fh);
}

function write_trial_data_csv($is_biased_choice, $current_reward, $current_unobserved_reward, $path){
    $file_name = $path . $_SESSION['user_id'] . '.csv';
    if (!file_exists($file_name)){
        file_put_contents($file_name,
            "trial_number, time, schedule_type, schedule_name, is_biased_choice, side_choice, RT, observed_reward, unobserved_reward, biased_reward, unbiased_reward" .PHP_EOL , FILE_APPEND);
    }
    global $current_biased_reward, $current_unbiased_reward;
    $trial_data =
        $_SESSION['trial_number'] . ','
        . date("Y-m-d h:i:sa") . ', '
        . $_SESSION['schedule_type'] . ', '
        . $_SESSION['schedule_name'] . ', '
        . strtolower($is_biased_choice) . ', '
        . $_GET['side']  . ', '
        . $_GET["RT"] . ', '
        . $current_reward . ', '
        . $current_unobserved_reward . ', '
        . $current_biased_reward . ', '
        . $current_unbiased_reward
        . PHP_EOL;
    file_put_contents($file_name, $trial_data, FILE_APPEND);
}
$path = "results/" . $_SESSION['schedule_type'] . "/" . $_SESSION['schedule_name'] . "/";
if (!file_exists($path)) { // Create the results directory for current mechanism if it doesn't exist yest
    mkdir($path, 0777, true);
}
write_trial_data_csv($is_biased_choice, $current_reward, $current_unobserved_reward, $path);

/////////////////////////////////////////////////////////////////////////////////
// Manage the game
/////////////////////////////////////////////////////////////////////////////////
if ($current_reward){
    $_SESSION["total_reward"]++;
}

$_SESSION['trial_number'] = $_SESSION['trial_number'] + 1;
echo $current_reward;
?>
