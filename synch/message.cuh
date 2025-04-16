#pragma once

enum MessageType {
    MSG_ACQUIRE,
    MSG_RELEASE,
    MSG_ACKNOWLEDGE
}

struct Message {
    MessageType MessageType;
    int sender_id;
    int data;
}