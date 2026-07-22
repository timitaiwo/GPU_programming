resource "runpod_pod" "h100_instella_pod" {
  compute_type = "GPU"

}

resource "runpod_network_volume" "weight_storer" {
  name = "weight-storer"
  data_center_id = "EUR-NO-2"
  size = 1000
}


// gpu type 
// name
// pod template - create custom
// count

// ssh_name 
// ssh_login command


// Network storage is S3 compatible so weights can be sent to the storage before the pod is started

